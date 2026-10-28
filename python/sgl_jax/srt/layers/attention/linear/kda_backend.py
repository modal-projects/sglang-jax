from __future__ import annotations

import dataclasses
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
from sgl_jax.srt.kernels.kda.verify import kda_multistep
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


@dataclasses.dataclass
class KDAVerifyRecord:
    """What a TARGET_VERIFY step needs to commit a KDA layer's state at the accepted position.

    A verify step runs T = draft_token_num tokens per request through the recurrence *without*
    writing the pool (the drafts may be rejected). Instead of checkpointing T full
    ``[H, K, V]`` states per request, it keeps the per-token recurrence inputs (k, v, gate,
    beta after conv/gate/l2norm: O(T * H * (2K + V)) per request) and the pre-verify state;
    :func:`commit_kda_verify` replays only the accepted prefix (masked, so shapes stay
    static) and scatters the result. The short-conv cache at the accepted position is the
    last K-1 raw inputs of ``conv0 ++ x[:accepted]``.
    """

    ssm_full: jax.Array  # [slots, H, K, V] pool buffer (unchanged by verify)
    conv_full: jax.Array  # [slots, proj, W] packed pool buffer (unchanged by verify)
    recurrent_indices: jax.Array  # [B]
    h0: jax.Array  # [B, H, K, V] pre-verify state
    conv0_q: jax.Array  # [B, Dq, W] pre-verify conv caches, per stream
    conv0_k: jax.Array
    conv0_v: jax.Array
    x_q: jax.Array  # [B*T, Dq] raw (pre-conv) projections, per stream
    x_k: jax.Array
    x_v: jax.Array
    k: jax.Array  # [B, T, H, K] l2-normalized keys
    v: jax.Array  # [B, T, H, V]
    g: jax.Array  # [B, T, H, K] activated log-space gate
    beta: jax.Array  # [B, T, H]
    mesh: jax.sharding.Mesh = dataclasses.field(metadata=dict(static=True), default=None)
    # SGLJAX_KDA_VERIFY_CHUNK=1 records: k/v are post-conv (pre-l2norm) and g is the RAW gate
    # (pre-activation), all flat [B*T, ...]; the commit runs the chunked prefill kernel.
    chunk: bool = dataclasses.field(metadata=dict(static=True), default=False)
    A_log: jax.Array | None = None  # [H]
    dt_bias: jax.Array | None = None  # [H, K]
    scale: float | None = dataclasses.field(metadata=dict(static=True), default=None)
    lower_bound: float | None = dataclasses.field(metadata=dict(static=True), default=None)


jax.tree_util.register_dataclass(KDAVerifyRecord)


def _scatter_rows(buf, idx, val):
    """``buf[idx] = val`` except rows with idx == 0 (padding / per-rank dummy slot)."""
    keep = (idx == 0).reshape((-1,) + (1,) * (buf.ndim - 1))
    return buf.at[idx].set(jnp.where(keep, buf[idx], val))


def commit_kda_verify(record: KDAVerifyRecord, num_tokens: jax.Array):
    """State after the first ``num_tokens[b]`` verify tokens of each request, scattered.

    ``num_tokens`` = committed tokens per request (the seed + accepted drafts, i.e.
    ``accept_len + 1`` in the DFlash worker), in ``[0, T]``. Returns the new
    ``(ssm_full, [conv_full])`` pool buffers for the layer.
    """
    mesh = record.mesh
    B = record.h0.shape[0]
    T = record.k.shape[0] // B if record.chunk else record.k.shape[1]
    W = record.conv0_q.shape[-1]
    # Committed tokens per request are in [0, T]; a larger count would read past the conv window.
    steps = jnp.clip(num_tokens.astype(jnp.int32), 0, T)
    if record.chunk:
        new_state = _commit_state_chunked(record, steps, T)
    else:
        new_state = _commit_state_naive(record, steps, T)
    return _commit_scatter(record, steps, T, W, new_state)


_NEG_GATE = -1.0e4  # raw gate for a masked step: activated gate -> 0 (no decay), exactly


def _commit_state_chunked(record, steps, T):
    """State after ``steps`` tokens via the chunked prefill kernel: non-live steps get k = v = 0,
    beta = 0 and a raw gate of -1e4 (activated gate 0) -- exact no-ops of the recurrence."""
    B = record.h0.shape[0]
    live = (jnp.arange(T, dtype=jnp.int32)[None, :] < steps[:, None]).reshape(-1)  # [B*T]
    k = jnp.where(live[:, None, None], record.k, jnp.zeros_like(record.k))
    v = jnp.where(live[:, None, None], record.v, jnp.zeros_like(record.v))
    g = jnp.where(live[:, None, None], record.g, jnp.full_like(record.g, _NEG_GATE))
    beta = jnp.where(live[:, None], record.beta, jnp.zeros_like(record.beta))
    cu = jnp.arange(B + 1, dtype=jnp.int32) * T
    dp = record.mesh.shape["data"]
    if dp > 1:  # per-DP-shard cumsum layout: [dp, B/dp + 1]
        per = B // dp
        cu = jnp.tile(jnp.arange(per + 1, dtype=jnp.int32) * T, dp)
    cu = jax.sharding.reshard(cu, jax.sharding.NamedSharding(record.mesh, P("data")))
    _, state = kda_extend_chunked(
        record.mesh, jnp.zeros_like(k), k, v, g, beta, record.h0, cu,
        record.A_log, record.dt_bias, record.scale, record.lower_bound,
    )
    return state


def _commit_state_naive(record, steps, T):
    mesh = record.mesh

    def _replay(k, v, g, beta, h0, steps):
        live = jnp.arange(T, dtype=jnp.int32)[None, :] < steps[:, None]  # [B, T]
        # A masked step is an exact no-op: exp(0) decay and a zero delta update.
        g = jnp.where(live[..., None, None], g, jnp.zeros_like(g))
        beta = jnp.where(live[..., None], beta, jnp.zeros_like(beta))
        # Zero k/v too: 0 * NaN from a rejected/padded step would still poison the state.
        k = jnp.where(live[..., None, None], k, jnp.zeros_like(k))
        v = jnp.where(live[..., None, None], v, jnp.zeros_like(v))
        if _VERIFY_KERNEL:
            _, state = kda_multistep(
                jnp.zeros_like(k), k, v, g, beta, h0, output_final_state=True
            )
        else:
            _, state = naive_recurrent_kda(
                jnp.zeros_like(k),
                k,
                v,
                g,
                beta,
                initial_state=h0,
                output_final_state=True,
            )
        return state.astype(h0.dtype)

    return jax.shard_map(
        _replay,
        mesh=mesh,
        in_specs=(
            P("data", None, "tensor", None),
            P("data", None, "tensor", None),
            P("data", None, "tensor", None),
            P("data", None, "tensor"),
            P("data", "tensor", None, None),
            P("data"),
        ),
        out_specs=P("data", "tensor", None, None),
        check_vma=False,
    )(record.k, record.v, record.g, record.beta, record.h0, steps)


def _commit_scatter(record, steps, T, W, new_state):
    mesh = record.mesh

    def _conv_window(c0, x, steps):
        b, d, _ = c0.shape
        x = jnp.swapaxes(x.reshape(b, T, d), 1, 2).astype(c0.dtype)  # [B, D, T]
        window = jnp.concatenate([c0, x], axis=-1)  # [B, D, W + T]
        idx = steps[:, None] + jnp.arange(W, dtype=jnp.int32)[None, :]
        return jnp.take_along_axis(window, idx[:, None, :], axis=-1)

    conv_window = jax.shard_map(
        lambda cq, ck, cv, xq, xk, xv, st: jnp.concatenate(
            [_conv_window(cq, xq, st), _conv_window(ck, xk, st), _conv_window(cv, xv, st)],
            axis=1,
        ),
        mesh=mesh,
        in_specs=(P("data", "tensor", None),) * 3 + (P("data", "tensor"),) * 3 + (P("data"),),
        out_specs=P("data", "tensor", None),
        check_vma=False,
    )
    new_conv = conv_window(
        record.conv0_q,
        record.conv0_k,
        record.conv0_v,
        record.x_q,
        record.x_k,
        record.x_v,
        steps,
    )

    ssm_full = jax.shard_map(
        _scatter_rows,
        mesh=mesh,
        in_specs=(P("data", "tensor", None, None), P("data"), P("data", "tensor", None, None)),
        out_specs=P("data", "tensor", None, None),
        check_vma=False,
    )(record.ssm_full, record.recurrent_indices, new_state)
    conv_full = jax.shard_map(
        _scatter_rows,
        mesh=mesh,
        in_specs=(P("data", "tensor", None), P("data"), P("data", "tensor", None)),
        out_specs=P("data", "tensor", None),
        check_vma=False,
    )(record.conv_full, record.recurrent_indices, new_conv)
    return ssm_full, [conv_full]


def commit_recurrent_verify(pool_updates: dict, num_tokens: jax.Array) -> dict:
    """Replace every :class:`KDAVerifyRecord` in a model's pool updates by committed buffers.

    No-op for models without recurrent layers or outside TARGET_VERIFY. Expects the
    ``{"recurrent_state_pool": (layers_recurrent, layers_conv)}`` layout the KDA models
    (Kimi-Linear, GLM-5.3) return.
    """
    rec = pool_updates.get("recurrent_state_pool") if isinstance(pool_updates, dict) else None
    if rec is None:
        return pool_updates
    layers_recurrent, layers_conv = rec
    if not any(isinstance(r, KDAVerifyRecord) for r in layers_recurrent):
        return pool_updates
    layers_recurrent, layers_conv = list(layers_recurrent), list(layers_conv)
    for i, r in enumerate(layers_recurrent):
        if isinstance(r, KDAVerifyRecord):
            layers_recurrent[i], layers_conv[i] = commit_kda_verify(r, num_tokens)
    updates = dict(pool_updates)
    updates["recurrent_state_pool"] = (layers_recurrent, layers_conv)
    return updates


# Opt-in (SGLJAX_KDA_CONV_DENSE=1): read/write the per-request conv states with one-hot matmuls over
# the whole (small) per-chip conv pool instead of an XLA row gather/scatter. The pool is
# [slots, proj/tp, K-1] (~2.4 MB per layer per chip at 65 slots); XLA:TPU's row scatter costs
# ~0.3 us per row (v6e-16 DP4 c192: 1.6 ms/step scatter + 0.3 ms gather over 34 KDA layers),
# while a dense rewrite of the pool is a few us. Exact: every output element has at most one
# non-zero term (slots are unique per request; idx 0 = padding, never written).
_CONV_DENSE = os.environ.get("SGLJAX_KDA_CONV_DENSE", "0") == "1"
# Opt-in: DFlash TARGET_VERIFY / commit through the chunked prefill kernel instead of the
# token-by-token naive recurrence (v8a 226 profile, c32 k8: naive verify + replay = ~35 of 88 ms).
_VERIFY_CHUNK = os.environ.get("SGLJAX_KDA_VERIFY_CHUNK", "0") == "1"
# Opt-in: DFlash verify + commit recurrence in a Pallas kernel with the state resident in VMEM across
# the T tokens (kernels/kda/verify.py). v8a 228 showed the chunked prefill kernel (above) is ~7x
# SLOWER at verify shapes (grid N x H x total-chunks, one chunk per 4-8-token request); this is the
# replacement. Same math and record layout as the naive path.
_VERIFY_KERNEL = os.environ.get("SGLJAX_KDA_VERIFY_KERNEL", "0") == "1"


def _prec(dtype):
    # bf16 x one-hot is exact at DEFAULT; wider state dtypes need the multi-pass f32 MXU path.
    return None if jnp.dtype(dtype).itemsize <= 2 else jax.lax.Precision.HIGHEST


def _select_einsum(eq, sel, x, prec):
    """einsum(eq, sel, x) for a 0/1 selection matrix ``sel``, NaN/Inf-safe: a non-finite value in a
    row that ``sel`` does not pick must not leak (0 * NaN = NaN in a plain matmul would poison every
    output). Non-finite inputs are zeroed for the value matmul and re-marked as NaN on the outputs
    that actually select them (a second, mask-only matmul)."""
    finite = jnp.isfinite(x)
    y = jnp.einsum(
        eq, sel, jnp.where(finite, x, jnp.zeros_like(x)), preferred_element_type=jnp.float32,
        precision=prec,
    )
    bad = jnp.einsum(eq, sel, (~finite).astype(sel.dtype), preferred_element_type=jnp.float32) > 0
    return jnp.where(bad, jnp.float32(jnp.nan), y)


def _dense_rows_get(buf, idx):
    # No flatten: a reshape of the [S, C, K-1] pool forces a relayout copy (XLA keeps C minor).
    slots = buf.shape[0]
    onehot = (idx[:, None] == jnp.arange(slots, dtype=idx.dtype)[None, :]).astype(buf.dtype)
    out = _select_einsum("bs,s...->b...", onehot, buf, _prec(buf.dtype))
    return out.astype(buf.dtype)


def _dense_rows_set(buf, idx, val, write):
    slots = buf.shape[0]
    hit = (idx[None, :] == jnp.arange(slots, dtype=idx.dtype)[:, None]) & write[None, :]  # [S, B]
    scattered = _select_einsum(
        "sb,b...->s...", hit.astype(buf.dtype), val.astype(buf.dtype), _prec(buf.dtype)
    )
    written = jnp.any(hit, axis=1).reshape((slots,) + (1,) * (buf.ndim - 1))
    return jnp.where(written, scattered.astype(buf.dtype), buf)


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


def kda_extend_chunked(mesh, q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias, scale,
                       lower_bound):
    """Packed varlen KDA prefill (Mega KDA with chunked fallback) under shard_map.

    q/k/v post-conv (l2norm in kernel), g the RAW gate (activation in kernel), beta post-sigmoid.
    Returns (o [T, H, V], final_state [N, H, K, V] in initial_state's dtype)."""
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
        mesh=mesh,
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
    # The prefill kernels take and return an FP32 state; a bf16 recurrent pool
    # (SGLANG_JAX_RECURRENT_STATE_DTYPE=bfloat16) is widened here and the final state is
    # rounded back to the pool dtype (the decode kernel does the same load/store in-kernel).
    state_dtype = initial_state.dtype
    o, final_state = sharded(
        q[None, ...],
        k[None, ...],
        v[None, ...],
        g[None, ...],
        beta[None, ...],
        initial_state.astype(jnp.float32),
        cu_seqlens,
        A_log,
        dt_bias,
    )
    return o[0], final_state.astype(state_dtype)



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
        is_verify = forward_batch.forward_mode == ForwardMode.TARGET_VERIFY
        x_q, x_k, x_v = q, k, v  # raw projections, kept for the verify commit
        if forward_batch.forward_mode == ForwardMode.EXTEND or is_verify:
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
        elif is_verify and _VERIFY_CHUNK:
            # Chunked-kernel verify: outputs from the prefill kernel over each request's T rows
            # (block-aligned cu_q_lens), no pool writes; commit reruns it on the accepted prefix.
            if layer.A_log is None or layer.dt_bias is None:
                raise ValueError("KDA gate activation requires layer.A_log and layer.dt_bias")
            H = q.shape[-2]
            A_log = layer.A_log.value.reshape(H)
            dt_bias = layer.dt_bias.value.reshape(H, -1)
            lower_bound = getattr(layer, "kda_gate_lower_bound", None)
            if lower_bound is None:
                lower_bound = getattr(layer, "kda_lower_bound", None)
            output, _ = kda_extend_chunked(
                self.mesh, q, k, v, a, b, ssm_states, cu_q_lens, A_log, dt_bias, layer.scale,
                lower_bound,
            )
            _, conv_full_list = self.get_layer_cache(recurrent_state_pool, layer.layer_id)
            ssm_full, _ = self.get_layer_cache(recurrent_state_pool, layer.layer_id)
            record = KDAVerifyRecord(
                ssm_full=ssm_full,
                conv_full=conv_full_list[0],
                recurrent_indices=recurrent_indices,
                h0=ssm_states,
                conv0_q=q_state,
                conv0_k=k_state,
                conv0_v=v_state,
                x_q=x_q,
                x_k=x_k,
                x_v=x_v,
                k=k,
                v=v,
                g=a,
                beta=b,
                mesh=self.mesh,
                chunk=True,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=None if layer.scale is None else float(layer.scale),
                lower_bound=None if lower_bound is None else float(lower_bound),
            )
            return output.reshape(output.shape[0], -1), (record, list(conv_full_list))
        elif is_verify:
            # Multi-token window per request (T = draft tokens), no pool writes: the commit
            # after acceptance replays the accepted prefix (commit_kda_verify).
            q = l2_normalize(q)
            k = l2_normalize(k)
            g = self._fused_kda_gate(layer, a)
            output, (k_b, v_b, g_b, beta_b) = self._forward_verify(
                q, k, v, g, b, ssm_states, scale=layer.scale
            )
            _, conv_full_list = self.get_layer_cache(recurrent_state_pool, layer.layer_id)
            ssm_full, _ = self.get_layer_cache(recurrent_state_pool, layer.layer_id)
            record = KDAVerifyRecord(
                ssm_full=ssm_full,
                conv_full=conv_full_list[0],
                recurrent_indices=recurrent_indices,
                h0=ssm_states,
                conv0_q=q_state,
                conv0_k=k_state,
                conv0_v=v_state,
                x_q=x_q,
                x_k=x_k,
                x_v=x_v,
                k=k_b,
                v=v_b,
                g=g_b,
                beta=beta_b,
                mesh=self.mesh,
            )
            return output.reshape(output.shape[0], -1), (record, list(conv_full_list))
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
            _dense_rows_get if _CONV_DENSE else (lambda buf, idx: buf[idx]),
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
            if _CONV_DENSE:
                return _dense_rows_set(buf, idx, val, idx != 0)
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
        return kda_extend_chunked(
            self.mesh, q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias, scale, lower_bound
        )

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

    @named_scope("kda_verify")
    def _forward_verify(self, q, k, v, g, beta, initial_state, scale=None):
        """T tokens per request (block-aligned rows) through the exact recurrence.

        Returns the per-token outputs [B*T, H, V] and the per-token recurrence inputs
        reshaped [B, T, ...] for :func:`commit_kda_verify`.
        """
        bsz = initial_state.shape[0]
        n = q.shape[0]
        if n % bsz:
            raise ValueError(f"KDA verify needs B*T rows: got {n} tokens for {bsz} requests.")
        T = n // bsz

        def _kernel(q, k, v, g, beta, h0):
            b_local = h0.shape[0]
            r = lambda x: x.reshape((b_local, T) + x.shape[1:])
            q, k, v, g, beta = map(r, (q, k, v, g, beta))
            if _VERIFY_KERNEL:
                o, _ = kda_multistep(q, k, v, g, beta, h0, scale=scale)
            else:
                o, _ = naive_recurrent_kda(
                    q, k, v, g, beta, scale=scale, initial_state=h0, output_final_state=False
                )
            return o.reshape((b_local * T,) + o.shape[2:]), k, v, g, beta

        tok = P("data", "tensor", None)
        blk = P("data", None, "tensor", None)
        o, k_b, v_b, g_b, beta_b = jax.shard_map(
            _kernel,
            mesh=self.mesh,
            in_specs=(tok, tok, tok, tok, P("data", "tensor"), P("data", "tensor", None, None)),
            out_specs=(tok, blk, blk, blk, P("data", None, "tensor")),
            check_vma=False,
        )(q, k, v, g, beta, initial_state)
        return o, (k_b, v_b, g_b, beta_b)

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


__all__ = [
    "KDAAttnBackend",
    "KDAVerifyRecord",
    "commit_kda_verify",
    "commit_recurrent_verify",
    "kda_decode_kernel_choice",
    "kda_decode_pool_sharded",
]
