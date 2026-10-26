from __future__ import annotations

import os
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.kda import (
    chunk_kda,
    is_mega_kda_layout_supported,
    kda_forward_packed,
    naive_recurrent_kda,
    resident_pipeline_kda_fwd,
)
from sgl_jax.srt.kernels.kda.decode import kda_decode_pool
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
)
from sgl_jax.srt.layers.attention.linear.short_convolution import short_convolution
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.profiling_utils import named_scope

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
    from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


def l2_normalize(x: jax.Array, epsilon: float = 1e-6) -> jax.Array:
    """Match FLA ``l2norm_fwd``: ``x / sqrt(sum(x**2) + eps)``."""
    x32 = x.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.sum(jnp.square(x32), axis=-1, keepdims=True) + epsilon)
    return (x32 * rstd).astype(x.dtype)


def kda_decode_kernel_choice() -> str:
    """``SGLANG_JAX_KDA_DECODE_KERNEL``: ``pallas`` | ``naive`` | ``auto`` (default).

    ``auto`` selects the in-place Pallas decode kernel on TPU and the naive JAX
    recurrence elsewhere (CPU tests, other backends).
    """
    choice = os.environ.get("SGLANG_JAX_KDA_DECODE_KERNEL", "auto").strip().lower()
    if choice == "auto":
        return "pallas" if jax.default_backend() == "tpu" else "naive"
    if choice not in ("pallas", "naive"):
        raise ValueError(
            f"SGLANG_JAX_KDA_DECODE_KERNEL must be 'pallas', 'naive' or 'auto', got {choice!r}"
        )
    return choice


class KDAAttnBackend(LinearRecurrentAttnBackend):
    """Attention backend for KDA (Kimi Delta Attention) linear attention."""

    def __init__(self, mesh: jax.sharding.Mesh = None):
        super().__init__(
            mesh=mesh,
        )

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        a: jax.Array,
        b: jax.Array,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        recurrent_state_pool: RecurrentStatePool,
        **kwargs,
    ) -> jax.Array:
        recurrent_indices = self.forward_metadata.recurrent_indices
        # Pallas decode reads/writes the SSM pool in place: no gathered copy.
        pallas_decode = (
            forward_batch.forward_mode == ForwardMode.DECODE
            and kda_decode_kernel_choice() == "pallas"
        )
        ssm_states, conv_states = self.get_state(
            recurrent_state_pool, layer.layer_id, recurrent_indices, skip_ssm=pallas_decode
        )
        q_conv_w = layer.q_conv1d.weight.value
        k_conv_w = layer.k_conv1d.weight.value
        v_conv_w = layer.v_conv1d.weight.value
        # _unpack_conv_states splits proj_size in 3 equal pieces; only valid
        # when proj_q == proj_k == proj_v (Kimi-Linear shape).
        assert (
            q_conv_w.shape[0] == k_conv_w.shape[0] == v_conv_w.shape[0]
        ), f"unequal Q/K/V proj widths: {q_conv_w.shape[0]}/{k_conv_w.shape[0]}/{v_conv_w.shape[0]}"
        q_state, k_state, v_state = self._unpack_conv_states(conv_states)

        cu_q_lens = self.forward_metadata.cu_q_lens
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            q, q_state_new = self._short_conv_extend(
                q, q_conv_w, q_state, cu_q_lens, layer.activation
            )
            k, k_state_new = self._short_conv_extend(
                k, k_conv_w, k_state, cu_q_lens, layer.activation
            )
            v, v_state_new = self._short_conv_extend(
                v, v_conv_w, v_state, cu_q_lens, layer.activation
            )
        else:
            q, q_state_new = short_convolution(
                q,
                q_conv_w,
                q_state,
                cu_q_lens,
                forward_batch.forward_mode,
                activation=layer.activation,
            )
            k, k_state_new = short_convolution(
                k,
                k_conv_w,
                k_state,
                cu_q_lens,
                forward_batch.forward_mode,
                activation=layer.activation,
            )
            v, v_state_new = short_convolution(
                v,
                v_conv_w,
                v_state,
                cu_q_lens,
                forward_batch.forward_mode,
                activation=layer.activation,
            )
        new_conv_packed = self._pack_conv_states(q_state_new, k_state_new, v_state_new)

        q = q.reshape(q.shape[0], layer.num_q_heads, layer.head_q_dim)
        k = k.reshape(k.shape[0], layer.num_k_heads, layer.head_k_dim)
        v = v.reshape(v.shape[0], layer.num_v_heads, layer.head_v_dim)
        a = a.reshape(a.shape[0], layer.num_q_heads, layer.head_q_dim)
        b = b.reshape(b.shape[0], layer.num_q_heads)

        if forward_batch.forward_mode == ForwardMode.EXTEND:
            output, new_recurrent = self._forward_extend(
                q,
                k,
                v,
                a,
                b,
                ssm_states,
                cu_q_lens,
                layer,
                scale=layer.scale,
            )
        elif forward_batch.forward_mode == ForwardMode.DECODE:
            # Decode uses the recurrent path (naive JAX or Pallas), so normalize
            # Q/K before the call. Mega KDA performs the same normalization
            # inside its prefill kernel; the chunked prefill fallback
            # normalizes in its branch.
            q = l2_normalize(q)
            k = l2_normalize(k)
            if pallas_decode:
                full_recurrent, _ = self.get_layer_cache(recurrent_state_pool, layer.layer_id)
                output, new_ssm_full = self._forward_decode_pallas(
                    q,
                    k,
                    v,
                    a,
                    b,
                    full_recurrent,
                    recurrent_indices,
                    layer,
                    scale=layer.scale,
                )
                new_recurrent = None
            else:
                output, new_recurrent = self._forward_decode(
                    q,
                    k,
                    v,
                    a,
                    b,
                    ssm_states,
                    layer,
                    scale=layer.scale,
                )
        else:
            raise NotImplementedError(f"KDA does not support {forward_batch.forward_mode}")

        if not pallas_decode:
            new_ssm_full = self.set_ssm_state(
                recurrent_state_pool, layer.layer_id, recurrent_indices, new_recurrent
            )
        new_conv_full_list = self.set_conv_state(
            recurrent_state_pool, layer.layer_id, recurrent_indices, new_conv_packed
        )

        # Track snapshot: boundaries are forced to forward ends, so the
        # running-scatter value IS the boundary snapshot; scatter it into each
        # request's track slot too. None -> byte-identical no-track path.
        track_indices = self.forward_metadata.recurrent_track_indices
        if track_indices is not None:
            track_mask = self.forward_metadata.recurrent_track_mask
            if not pallas_decode:  # the Pallas kernel already wrote the track slots
                new_ssm_full = self.set_ssm_track_state(
                    new_ssm_full, track_indices, track_mask, new_recurrent
                )
            new_conv_full_list = self.set_conv_track_state(
                new_conv_full_list, track_indices, track_mask, new_conv_packed
            )

        return output.reshape(output.shape[0], -1), (new_ssm_full, new_conv_full_list)

    def get_state(self, recurrent_state_pool, layer_id, recurrent_indices, skip_ssm=False):
        """Return per-request views of (ssm, conv) state for this layer.

        ``skip_ssm``: return ``None`` for ssm (the caller reads the pool in place).
        """
        recurrent_buffer, conv_buffer_list = self.get_layer_cache(
            recurrent_state_pool,
            layer_id,
        )
        assert len(conv_buffer_list) == 1, (
            f"KDA expects exactly 1 conv buffer per layer "
            f"(reserved RecurrentStatePool inner-list length); got {len(conv_buffer_list)}"
        )
        conv_buffer = conv_buffer_list[0]

        ssm = None
        if not skip_ssm:
            ssm = jax.shard_map(
                lambda buf, idx: buf[idx],
                mesh=self.mesh,
                in_specs=(P("data", "tensor", None, None), P("data")),
                out_specs=P("data", "tensor", None, None),
                check_vma=False,
            )(recurrent_buffer, recurrent_indices)

        conv = jax.shard_map(
            lambda buf, idx: buf[idx],
            mesh=self.mesh,
            in_specs=(P("data", "tensor", None), P("data")),
            out_specs=P("data", "tensor", None),
            check_vma=False,
        )(conv_buffer, recurrent_indices)

        has_initial_state = self.forward_metadata.has_initial_state
        if has_initial_state is not None:
            if ssm is not None:
                ssm = jnp.where(has_initial_state[:, None, None, None], ssm, 0.0)
            conv = jnp.where(has_initial_state[:, None, None], conv, 0.0)

        return ssm, conv

    def set_ssm_state(self, recurrent_state_pool, layer_id, recurrent_indices, new_recurrent):
        """Scatter per-request ``new_recurrent`` into the FULL pool buffer.

        Suppress writes at idx==0: padding rows carry idx=0 and would otherwise
        pollute the per-rank dummy slot, leaking garbage back as initial state.
        """
        full_recurrent, _ = self.get_layer_cache(recurrent_state_pool, layer_id)

        def _scatter(buf, idx, val):
            keep_mask = (idx == 0).reshape(-1, 1, 1, 1)
            safe_val = jnp.where(keep_mask, buf[idx], val)
            return buf.at[idx].set(safe_val)

        return jax.shard_map(
            _scatter,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None, None),
                P("data"),
                P("data", "tensor", None, None),
            ),
            out_specs=P("data", "tensor", None, None),
            check_vma=False,
        )(full_recurrent, recurrent_indices, new_recurrent)

    def set_conv_state(self, recurrent_state_pool, layer_id, recurrent_indices, new_conv_packed):
        """Scatter per-request packed conv state. Same idx==0 guard as set_ssm_state."""
        _, conv_buffer_list = self.get_layer_cache(recurrent_state_pool, layer_id)
        assert len(conv_buffer_list) == 1
        full_conv = conv_buffer_list[0]

        def _scatter(buf, idx, val):
            keep_mask = (idx == 0).reshape(-1, 1, 1)
            safe_val = jnp.where(keep_mask, buf[idx], val)
            return buf.at[idx].set(safe_val)

        new_conv_full = jax.shard_map(
            _scatter,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None),
                P("data"),
                P("data", "tensor", None),
            ),
            out_specs=P("data", "tensor", None),
            check_vma=False,
        )(full_conv, recurrent_indices, new_conv_packed)
        return [new_conv_full]

    def set_ssm_track_state(self, full_recurrent, track_indices, track_mask, new_recurrent):
        """Scatter ``new_recurrent`` into the request track slots.

        Operates on the buffer RETURNED by ``set_ssm_state`` so both the
        running slot and the track slot persist. Keep-mask preserves slots at
        ``track_idx == 0`` (padding/dummy) and ``track_mask == 0`` (non-boundary
        rows). Wrapped in ``optimization_barrier`` because the buffer is donated
        (donated-buffer multi-host aliasing guard).
        """

        def _scatter(buf, tidx, tmask, val):
            keep = ((tidx == 0) | (tmask == 0)).reshape(-1, 1, 1, 1)
            safe_val = jnp.where(keep, buf[tidx], val)
            return buf.at[tidx].set(safe_val)

        new_full = jax.shard_map(
            _scatter,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None, None),
                P("data"),
                P("data"),
                P("data", "tensor", None, None),
            ),
            out_specs=P("data", "tensor", None, None),
            check_vma=False,
        )(full_recurrent, track_indices, track_mask, new_recurrent)
        return jax.lax.optimization_barrier(new_full)

    def set_conv_track_state(self, new_conv_full_list, track_indices, track_mask, new_conv_packed):
        """Scatter ``new_conv_packed`` into the request track slots.

        Conv variant of :meth:`set_ssm_track_state` (3D keep-mask reshape).
        """
        full_conv = new_conv_full_list[0]

        def _scatter(buf, tidx, tmask, val):
            keep = ((tidx == 0) | (tmask == 0)).reshape(-1, 1, 1)
            safe_val = jnp.where(keep, buf[tidx], val)
            return buf.at[tidx].set(safe_val)

        new_full = jax.shard_map(
            _scatter,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None),
                P("data"),
                P("data"),
                P("data", "tensor", None),
            ),
            out_specs=P("data", "tensor", None),
            check_vma=False,
        )(full_conv, track_indices, track_mask, new_conv_packed)
        return [jax.lax.optimization_barrier(new_full)]

    # ------------------------------------------------------------------
    # Forward mode implementations
    # ------------------------------------------------------------------

    def _short_conv_extend(
        self,
        x: jax.Array,
        weight: jax.Array,
        cache: jax.Array,
        cu_seqlens: jax.Array,
        activation,
    ) -> tuple[jax.Array, jax.Array]:
        """EXTEND-path conv wrapped in shard_map."""

        def _call(x, weight, cache, cu_seqlens):
            return short_convolution(
                x, weight, cache, cu_seqlens, ForwardMode.EXTEND, activation=activation
            )

        sharded = jax.shard_map(
            _call,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor"),  # x [T, hidden]
                P("tensor", None),  # weight [hidden, K]
                P("data", "tensor", None),  # cache [B, hidden, K-1]
                P("data"),  # cu_seqlens [B+1]
            ),
            out_specs=(
                P("data", "tensor"),  # y [T, hidden]
                P("data", "tensor", None),  # new_cache [B, hidden, K-1]
            ),
            check_vma=False,
        )
        return sharded(x, weight, cache, cu_seqlens)

    @named_scope("kda_extend")
    def _forward_extend(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        initial_state: jax.Array,
        cu_seqlens: jax.Array,
        layer: RadixLinearAttention,
        scale: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Prefill via Mega KDA, with a guarded fallback to the chunked kernel."""
        if layer.A_log is None or layer.dt_bias is None:
            raise ValueError("KDA gate activation requires layer.A_log and layer.dt_bias")
        H = q.shape[-2]
        # kda_gate_chunk_cumsum requires A_log shape (H,); layer stores [1,1,H,1].
        A_log = layer.A_log.value.reshape(H)
        dt_bias = layer.dt_bias.value.reshape(H, -1)
        scale = scale if scale is not None else layer.scale
        lower_bound = getattr(layer, "kda_gate_lower_bound", None)
        if lower_bound is None:
            lower_bound = getattr(layer, "kda_lower_bound", None)
        kernel = os.environ.get("SGLANG_JAX_KDA_PREFILL_KERNEL", "mega").strip().lower()
        if kernel not in {"chunked", "mega", "resident", "fused_resident"}:
            raise ValueError(
                "SGLANG_JAX_KDA_PREFILL_KERNEL must be 'chunked', 'mega', or 'resident', "
                f"got {kernel!r}"
            )
        use_mega = kernel == "mega"
        use_resident = kernel in {"resident", "fused_resident"}

        def _prefill_call(q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias):
            operands = (q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias)

            if use_resident:
                o, final_state, *_ = resident_pipeline_kda_fwd(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    scale,
                    initial_state,
                    True,
                    cu_seqlens,
                    safe_gate=True,
                    lower_bound=lower_bound,
                    use_gate_in_kernel=True,
                    use_qk_l2norm_in_kernel=True,
                    A_log=A_log,
                    dt_bias=dt_bias,
                )
                return o, final_state

            def _chunked(args):
                q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias = args
                q = l2_normalize(q)
                k = l2_normalize(k)
                o, final_state, *_ = chunk_kda(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    scale=scale,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    use_gate_in_kernel=True,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    lower_bound=lower_bound,
                    pipeline_impl="incumbent",
                )
                return o, final_state

            if not use_mega:
                return _chunked(operands)

            def _mega(args):
                q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias = args
                return kda_forward_packed(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu_seqlens=cu_seqlens,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    scale=scale,
                    initial_state=initial_state,
                    lower_bound=lower_bound,
                    pipeline_impl="incumbent",
                )

            padded_tokens = (q.shape[1] + 63) // 64 * 64
            layout_supported = is_mega_kda_layout_supported(
                cu_seqlens,
                padded_tokens,
            )
            return jax.lax.cond(layout_supported, _mega, _chunked, operands)

        sharded = jax.shard_map(
            _prefill_call,
            mesh=self.mesh,
            in_specs=(
                P(None, "data", "tensor", None),  # q [1, T, H, K]
                P(None, "data", "tensor", None),  # k [1, T, H, K]
                P(None, "data", "tensor", None),  # v [1, T, H, V]
                P(None, "data", "tensor", None),  # g [1, T, H, K]
                P(None, "data", "tensor"),  # beta [1, T, H]
                P("data", "tensor", None, None),  # initial_state [N, H, K, V]
                P("data"),  # cu_seqlens [N+1]
                P("tensor"),  # A_log [H]
                P("tensor", None),  # dt_bias [H, K]
            ),
            out_specs=(
                P(None, "data", "tensor", None),  # output [1, T, H, V]
                P("data", "tensor", None, None),  # final_state [N, H, K, V]
            ),
            check_vma=False,
        )
        # Kernel expects [1, T_packed, H, K] packed layout; strip after.
        o, final_state = sharded(
            q[None, ...],
            k[None, ...],
            v[None, ...],
            g[None, ...],
            beta[None, ...],
            initial_state,
            cu_seqlens,
            A_log,
            dt_bias,
        )
        return o[0], final_state

    @named_scope("kda_decode")
    def _forward_decode(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        initial_state: jax.Array,
        layer: RadixLinearAttention,
        scale: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Single-step decode via naive JAX recurrence (Pallas decode TBD)."""
        g = self._fused_kda_gate(layer, g)

        def _decode_kernel(q_d, k_d, v_d, g_d, beta_d, h0):
            o, final_state = naive_recurrent_kda(
                q_d[:, None, ...],
                k_d[:, None, ...],
                v_d[:, None, ...],
                g_d[:, None, ...],
                beta_d[:, None, ...],
                scale=scale,
                initial_state=h0,
                output_final_state=True,
            )
            return o[:, 0], final_state

        sharded = jax.shard_map(
            _decode_kernel,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None),  # q [B, H, K]
                P("data", "tensor", None),  # k [B, H, K]
                P("data", "tensor", None),  # v [B, H, V]
                P("data", "tensor", None),  # g [B, H, K]
                P("data", "tensor"),  # beta [B, H]
                P("data", "tensor", None, None),  # h0 [B, H, K, V]
            ),
            out_specs=(
                P("data", "tensor", None),  # o [B, H, V]
                P("data", "tensor", None, None),  # final_state [B, H, K, V]
            ),
            check_vma=False,
        )
        return sharded(q, k, v, g, beta, initial_state)

    @named_scope("kda_decode_pallas")
    def _forward_decode_pallas(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        full_recurrent: jax.Array,
        recurrent_indices: jax.Array,
        layer: RadixLinearAttention,
        scale: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Single-step decode with the Pallas kernel, in place on the SSM pool.

        Same math and slot semantics as ``get_state`` -> ``_forward_decode`` ->
        ``set_ssm_state`` (-> ``set_ssm_track_state``), but each request's state
        is DMA'd straight from / to its pool slot (pool is input/output aliased),
        so the [B, H, K, V] state is read once and written once.

        Returns (o [B, H, V], new full pool buffer).
        """
        g = self._fused_kda_gate(layer, g)
        md = self.forward_metadata
        return kda_decode_pool_sharded(
            self.mesh,
            q,
            k,
            v,
            g,
            beta,
            full_recurrent,
            recurrent_indices,
            scale=scale,
            has_initial_state=md.has_initial_state,
            track_indices=md.recurrent_track_indices,
            track_mask=md.recurrent_track_mask,
        )

    def _fused_kda_gate(
        self,
        layer: RadixLinearAttention,
        g: jax.Array,
    ) -> jax.Array:
        """JAX-side gate activation used by the DECODE path."""
        if layer.A_log is None or layer.dt_bias is None:
            raise ValueError("KDA gate activation requires layer.A_log and layer.dt_bias")
        H = g.shape[-2]
        orig_dtype = g.dtype
        g32 = g.astype(jnp.float32) + layer.dt_bias.value.reshape(H, -1).astype(jnp.float32)
        a_scale = jnp.exp(layer.A_log.value.reshape(H, 1).astype(jnp.float32))
        lower_bound = getattr(layer, "kda_gate_lower_bound", None)
        if lower_bound is None:
            lower_bound = getattr(layer, "kda_lower_bound", None)
        if lower_bound is None:
            out = -a_scale * jax.nn.softplus(g32)
        else:
            out = lower_bound * jax.nn.sigmoid(a_scale * g32)
        return out.astype(orig_dtype)

    def _unpack_conv_states(
        self,
        conv_states: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Slice ``[B, proj_size, K-1]`` into per-stream Q/K/V caches.

        The packed buffer is laid out per tensor shard: shard ``r`` holds
        ``[q_r | k_r | v_r]``, where ``q_r`` is the shard's slice of the Q
        stream. Splitting inside ``shard_map`` keeps each stream sharded like
        its conv weight and activations. A global ``jnp.split`` of a
        ``[q | k | v]`` buffer would cross shard boundaries whenever
        ``proj_size / tp`` does not align with the stream width, and XLA
        would reshard every layer, every step.
        """
        D = conv_states.shape[1]
        assert D % 3 == 0, f"conv_states channel dim {D} must be divisible by 3"
        spec = P("data", "tensor", None)
        return jax.shard_map(
            lambda c: tuple(jnp.split(c, 3, axis=1)),
            mesh=self.mesh,
            in_specs=(spec,),
            out_specs=(spec, spec, spec),
            check_vma=False,
        )(conv_states)

    def _pack_conv_states(
        self,
        q_state: jax.Array,
        k_state: jax.Array,
        v_state: jax.Array,
    ) -> jax.Array:
        """Concat per-stream ``[B, D, K-1]`` caches → packed ``[B, proj_size, K-1]``.

        Inverse of :meth:`_unpack_conv_states`: concatenates per shard.
        """
        spec = P("data", "tensor", None)
        return jax.shard_map(
            lambda q, k, v: jnp.concatenate([q, k, v], axis=1),
            mesh=self.mesh,
            in_specs=(spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )(q_state, k_state, v_state)


def kda_decode_pool_sharded(
    mesh,
    q,
    k,
    v,
    g,
    beta,
    full_recurrent,
    recurrent_indices,
    *,
    scale=None,
    has_initial_state=None,
    track_indices=None,
    track_mask=None,
    **kernel_kwargs,
):
    """``kda_decode_pool`` under ``shard_map`` over ("data", "tensor").

    Pool [S, H, K, V] is P("data", "tensor", None, None) with per-DP-rank local
    slot indices (slot 0 = dummy), matching ``RecurrentStatePool``.
    """
    has_track = track_indices is not None
    if has_track and track_mask is None:
        track_mask = jnp.ones_like(track_indices)

    def _call(q, k, v, g, beta, pool, idx, *meta):
        meta = list(meta)
        hi = meta.pop(0) if has_initial_state is not None else None
        tidx, tmask = (meta[0], meta[1]) if has_track else (None, None)
        return kda_decode_pool(
            q,
            k,
            v,
            g,
            beta,
            pool,
            idx,
            scale=scale,
            has_initial_state=hi,
            track_idx=tidx,
            track_mask=tmask,
            **kernel_kwargs,
        )

    meta = []
    if has_initial_state is not None:
        meta.append(has_initial_state)
    if has_track:
        meta += [track_indices, track_mask]
    o, new_pool = jax.shard_map(
        _call,
        mesh=mesh,
        in_specs=(
            P("data", "tensor", None),  # q [B, H, K]
            P("data", "tensor", None),  # k [B, H, K]
            P("data", "tensor", None),  # v [B, H, V]
            P("data", "tensor", None),  # g [B, H, K]
            P("data", "tensor"),  # beta [B, H]
            P("data", "tensor", None, None),  # pool [S, H, K, V]
            P("data"),  # recurrent_indices [B]
            *([P("data")] * len(meta)),
        ),
        out_specs=(
            P("data", "tensor", None),  # o [B, H, V]
            P("data", "tensor", None, None),  # pool
        ),
        check_vma=False,
    )(q, k, v, g, beta, full_recurrent, recurrent_indices, *meta)
    if has_track:
        # Same donated-buffer aliasing guard as set_ssm_track_state.
        new_pool = jax.lax.optimization_barrier(new_pool)
    return o, new_pool


__all__ = ["KDAAttnBackend", "kda_decode_kernel_choice", "kda_decode_pool_sharded"]
