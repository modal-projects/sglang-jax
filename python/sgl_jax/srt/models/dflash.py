from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.layers.radix_attention import AttentionType, RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.speculative.dflash_util import (
    DFlashModelOptions,
    get_dflash_model_options,
)
from sgl_jax.srt.utils.profiling_utils import named_scope

logger = logging.getLogger(__name__)


def get_dflash_layer_attention_params(
    config: PretrainedConfig, layer_id: int
) -> tuple[int, AttentionType]:
    """(sliding_window_size, attention type) for one draft layer, as upstream SGLang decides.

    Without ``layer_types`` the draft block attends bidirectionally (classic DFlash). With them,
    ``full_attention`` defaults to bidirectional and ``sliding_attention`` to causal, and an
    explicit ``is_causal`` overrides either. The window is the HF one (it includes the query
    token), which is what the JAX ragged kernels expect (upstream torch stores window - 1).
    """
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return 0, AttentionType.ENCODER_ONLY
    if layer_id >= len(layer_types):
        raise ValueError(
            "DFLASH config.layer_types must contain one entry per draft layer, "
            f"got {len(layer_types)} entries for layer_id={layer_id}."
        )
    layer_type = layer_types[layer_id]
    is_causal = getattr(config, "is_causal", None)
    if layer_type == "full_attention":
        window, default = 0, AttentionType.ENCODER_ONLY
    elif layer_type == "sliding_attention":
        window = getattr(config, "sliding_window", None)
        if window is None:
            raise ValueError("DFLASH sliding_attention layers require config.sliding_window.")
        window, default = int(window), AttentionType.DECODER
    else:
        raise ValueError(
            f"Unsupported DFLASH draft layer type layer_types[{layer_id}]={layer_type!r}."
        )
    if is_causal is None:
        return window, default
    return window, (AttentionType.DECODER if is_causal else AttentionType.ENCODER_ONLY)


class DFlashAttention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = int(config.hidden_size)
        self.head_dim = int(
            getattr(config, "head_dim", self.hidden_size // config.num_attention_heads)
        )
        self.q_head_num = int(config.num_attention_heads)
        self.kv_head_num = int(getattr(config, "num_key_value_heads", self.q_head_num))
        self.q_size = self.q_head_num * self.head_dim
        self.kv_size = self.kv_head_num * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.mesh = mesh

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.q_norm = RMSNorm(
            self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_norm"
        )
        self.k_norm = RMSNorm(
            self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="k_norm"
        )

        self.q_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.q_size,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.kv_size,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.kv_size,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="v_proj",
        )
        self.o_proj = LinearBase(
            input_size=self.q_size,
            output_size=self.hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )

        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=int(getattr(config, "max_position_embeddings", 32768)),
            base=float(config.rope_parameters["rope_theta"]),
            is_neox_style=True,
            dtype=dtype,
        )
        sliding_window_size, attn_type = get_dflash_layer_attention_params(config, layer_id)
        self.attn = RadixAttention(
            num_heads=self.q_head_num,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.kv_head_num,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            attn_type=attn_type,
        )

    @named_scope
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(
            -1,
            self.q_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        k = k.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        v = v.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        output, _ = self.o_proj(attn_output)
        return output, kv_fused

    def kv_proj(
        self, positions: jax.Array, hidden_states: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        num_tokens = positions.shape[0]
        hidden_states = hidden_states[:num_tokens]
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        k = k.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        v = v.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        k = self.k_norm(k)
        dummy_q = jnp.zeros((num_tokens, 1, self.head_dim), dtype=k.dtype)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k, v


class DFlashMLP(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="gate_proj",
        )
        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="up_proj",
        )
        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="down_proj",
        )
        self.act_fn = jax.nn.silu

    @named_scope
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        hidden_states = self.act_fn(gate) * up
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class DFlashGroupedConv(nnx.Module):
    """DFlash2 grouped dynamic depthwise K-tap convolution across one draft block.

    Port of upstream ``DFlashGroupedConv``. One projection of the sublayer *input* yields two
    per-token, per-group kernels: ``prepare`` convolves the input with the first, ``finish``
    convolves the sublayer output with the second. Tap ``t`` mixes the token ``t`` positions
    earlier in the same block (causal within the block, zero across block boundaries):

        coeff[n, t, c] = base_kernel[side, t, c] + delta[n, side, t, c // group_size]
        out[n, c]      = sum_t coeff[n, t, c] * x[n - t, c] * [n % block_size >= t]

    The flattened token axis must be block-aligned (every request contributes exactly
    ``block_size`` consecutive rows), as in the DFlash draft forward.
    """

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        taps: int,
        group_size: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        if hidden_size % group_size:
            raise ValueError(
                f"DFLASH conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = self.hidden_size // self.group_size
        self.mesh = mesh
        # [input/output side, tap, channel], the layout training exports. Identity at init.
        base = jnp.zeros(
            (2, self.taps, self.hidden_size),
            dtype=dtype,
            out_sharding=NamedSharding(mesh, P(None, None, None)),
        )
        self.base_kernel = nnx.Param(base.at[:, 0].set(1))
        self.kernel_projection = LinearBase(
            input_size=self.hidden_size,
            output_size=2 * self.taps * self.num_groups,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="kernel_projection",
        )

    def _convolve(self, hidden_states: jax.Array, delta: jax.Array, side: int) -> jax.Array:
        """hidden_states [N, H], delta [N, taps, G] -> [N, H]."""
        n, h = hidden_states.shape
        bsz, taps = self.block_size, self.taps
        if n % bsz:
            raise ValueError(f"DFLASH conv needs block-aligned tokens: N={n}, block_size={bsz}.")
        rows = NamedSharding(self.mesh, P("data", None))
        x = jax.sharding.reshard(hidden_states, rows)
        # Per-channel coefficients: expand each group's delta over its group_size channels.
        delta = jax.sharding.reshard(delta.reshape(n, taps * self.num_groups), rows)
        delta = jnp.repeat(
            delta.reshape(
                n,
                taps,
                self.num_groups,
                out_sharding=NamedSharding(self.mesh, P("data", None, None)),
            ),
            self.group_size,
            axis=-1,
            out_sharding=NamedSharding(self.mesh, P("data", None, None)),
        )
        coeff = self.base_kernel.value[side][None].astype(delta.dtype) + delta  # [N, taps, H]
        blk = NamedSharding(self.mesh, P("data", None, None))
        xb = x.reshape(n // bsz, bsz, h, out_sharding=blk)
        cb = coeff.reshape(
            n // bsz,
            bsz,
            taps,
            h,
            out_sharding=NamedSharding(self.mesh, P("data", None, None, None)),
        )
        out = cb[:, :, 0] * xb
        for tap in range(1, taps):
            # Shift inside each block; rows with in-block position < tap read zeros.
            shifted = jnp.pad(xb[:, : bsz - tap], ((0, 0), (tap, 0), (0, 0)))
            out = out + cb[:, :, tap] * shifted
        return out.reshape(n, h, out_sharding=rows).astype(hidden_states.dtype)

    def prepare(self, hidden_states: jax.Array) -> tuple[jax.Array, jax.Array]:
        coefficients, _ = self.kernel_projection(
            hidden_states, out_sharding=NamedSharding(self.mesh, P("data", None))
        )
        n = hidden_states.shape[0]
        coefficients = coefficients.reshape(
            n,
            2,
            self.taps,
            self.num_groups,
            out_sharding=NamedSharding(self.mesh, P("data", None, None, None)),
        )
        return (
            self._convolve(hidden_states, coefficients[:, 0], side=0),
            coefficients[:, 1],
        )

    def finish(self, hidden_states: jax.Array, coefficients: jax.Array) -> jax.Array:
        return self._convolve(hidden_states, coefficients, side=1)


class DFlashDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        options: DFlashModelOptions | None = None,
    ):
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        if options is not None and options.has_conv:

            def grouped_conv():
                return DFlashGroupedConv(
                    hidden_size,
                    options.block_size,
                    options.conv_kernel_size,
                    options.conv_group_size,
                    mesh=mesh,
                    dtype=dtype,
                )

            self.attention_conv = grouped_conv()
            self.mlp_conv = grouped_conv()
        else:
            self.attention_conv = None
            self.mlp_conv = None
        self.input_layernorm = RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scope_name="input_layernorm",
        )
        self.self_attn = DFlashAttention(config=config, mesh=mesh, layer_id=layer_id, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_attention_layernorm",
        )
        self.mlp = DFlashMLP(config=config, mesh=mesh, layer_id=layer_id, dtype=dtype)

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        attention_kernel = None
        if self.attention_conv is not None:
            hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)
        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )
        if attention_kernel is not None:
            hidden_states = self.attention_conv.finish(hidden_states, attention_kernel)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_kernel = None
        if self.mlp_conv is not None:
            hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if mlp_kernel is not None:
            hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        return hidden_states, residual, kv_fused


class DFlashBackbone(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        options: DFlashModelOptions | None = None,
    ):
        self.layers = nnx.data(
            [
                DFlashDecoderLayer(
                    config=config, mesh=mesh, layer_id=i, dtype=dtype, options=options
                )
                for i in range(int(config.num_hidden_layers))
            ]
        )
        self.norm = RMSNorm(
            int(config.hidden_size),
            epsilon=float(getattr(config, "rms_norm_eps", 1e-6)),
            param_dtype=dtype,
            scope_name="norm",
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, list[jax.Array]]:
        hidden_states = forward_batch.input_embedding
        if hidden_states is None:
            raise ValueError("DFlashDraftModel requires forward_batch.input_embedding.")

        residual = None
        layers_kv_fused = []
        for layer in self.layers:
            hidden_states, residual, kv_fused = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)
        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused


class DFlashDraftModel(nnx.Module):
    """DFlash draft transformer without owned embedding or LM head."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.options = self._model_options(config)
        self.model = DFlashBackbone(config=config, mesh=mesh, dtype=dtype, options=self.options)

        hidden_size = int(config.hidden_size)
        target_hidden_size = int(getattr(config, "target_hidden_size", hidden_size))
        target_layer_ids = getattr(config, "target_layer_ids", None)
        if target_layer_ids is None:
            dflash_config = getattr(config, "dflash_config", {}) or {}
            target_layer_ids = dflash_config.get("target_layer_ids", None)
        num_context_features = (
            len(target_layer_ids) if target_layer_ids is not None else int(config.num_hidden_layers)
        )
        self.num_context_features = int(num_context_features)
        self.target_hidden_size = target_hidden_size

        self.fc = LinearBase(
            input_size=self.num_context_features * self.target_hidden_size,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="fc",
        )
        self.hidden_norm = RMSNorm(
            hidden_size,
            epsilon=float(getattr(config, "rms_norm_eps", 1e-6)),
            param_dtype=dtype,
            scope_name="hidden_norm",
        )

    @staticmethod
    def _model_options(config: PretrainedConfig) -> DFlashModelOptions | None:
        # Classic DFlash checkpoints never carry conv/selector weights; parsing is only needed
        # (and validated) for DFlash2.
        return None

    def project_target_hidden(self, target_hidden: jax.Array) -> jax.Array:
        expected = self.num_context_features * self.target_hidden_size
        if target_hidden.shape[-1] != expected:
            raise ValueError(
                "DFLASH target hidden feature mismatch: "
                f"expected last dim {expected}, got {target_hidden.shape[-1]}."
            )
        hidden_states, _ = self.fc(target_hidden)
        return self.hidden_norm(hidden_states)

    def materialize_kv(
        self,
        target_hidden: jax.Array,
        positions: jax.Array,
    ) -> list[tuple[jax.Array, jax.Array]]:
        ctx = self.project_target_hidden(target_hidden)
        return [layer.self_attn.kv_proj(positions, ctx) for layer in self.model.layers]

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: Any = None,
    ):
        hidden_states, layers_kv_fused = self.model(
            forward_batch,
            memory_pools.token_to_kv_pool,
        )
        output = LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden_states)
        return output, {"token_to_kv_pool": layers_kv_fused}, [], None

    def sample_block_tokens(
        self, base_logits: jax.Array, first_prev_tokens: jax.Array
    ) -> jax.Array:
        return jnp.argmax(base_logits, axis=-1).astype(jnp.int32)

    def load_weights(self, model_config: ModelConfig) -> None:
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load(self._create_weight_mappings())
        logger.info("DFlash draft weights loaded successfully.")

    def _create_weight_mappings(self) -> dict[str, WeightSpec]:
        mappings: dict[str, WeightSpec] = {
            "fc.weight": WeightSpec(
                target_path="fc.weight",
                sharding=(None, None),
                transpose=True,
            ),
            "hidden_norm.weight": WeightSpec(
                target_path="hidden_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            "norm.weight": WeightSpec(
                target_path="model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        for layer_idx in range(int(self.config.num_hidden_layers)):
            prefix = f"layers.{layer_idx}"
            target = f"model.layers.{layer_idx}"
            mappings.update(
                {
                    f"{prefix}.input_layernorm.weight": WeightSpec(
                        target_path=f"{target}.input_layernorm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.post_attention_layernorm.weight": WeightSpec(
                        target_path=f"{target}.post_attention_layernorm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.self_attn.q_proj.weight": WeightSpec(
                        target_path=f"{target}.self_attn.q_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.k_proj.weight": WeightSpec(
                        target_path=f"{target}.self_attn.k_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                        kv_head_padding=True,
                    ),
                    f"{prefix}.self_attn.v_proj.weight": WeightSpec(
                        target_path=f"{target}.self_attn.v_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                        kv_head_padding=True,
                    ),
                    f"{prefix}.self_attn.o_proj.weight": WeightSpec(
                        target_path=f"{target}.self_attn.o_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.q_norm.weight": WeightSpec(
                        target_path=f"{target}.self_attn.q_norm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.self_attn.k_norm.weight": WeightSpec(
                        target_path=f"{target}.self_attn.k_norm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.mlp.gate_proj.weight": WeightSpec(
                        target_path=f"{target}.mlp.gate_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.mlp.up_proj.weight": WeightSpec(
                        target_path=f"{target}.mlp.up_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.mlp.down_proj.weight": WeightSpec(
                        target_path=f"{target}.mlp.down_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                    ),
                }
            )
            if getattr(self.config, "attention_bias", False):
                for proj, kv_padding in (
                    ("q_proj", False),
                    ("k_proj", True),
                    ("v_proj", True),
                ):
                    mappings[f"{prefix}.self_attn.{proj}.bias"] = WeightSpec(
                        target_path=f"{target}.self_attn.{proj}.bias",
                        sharding=(None,),
                        transpose=False,
                        kv_head_padding=kv_padding,
                    )
                mappings[f"{prefix}.self_attn.o_proj.bias"] = WeightSpec(
                    target_path=f"{target}.self_attn.o_proj.bias",
                    sharding=(None,),
                    transpose=False,
                )

        return mappings


def candidate_topk(
    logits: jax.Array, k: int, *, vocab_size: int, mesh: jax.sharding.Mesh
) -> tuple[jax.Array, jax.Array]:
    """Exact top-k over a (possibly vocab-sharded, padded) logits matrix [N, V_pad].

    Mirrors upstream ``candidate_topk`` under TP: a local top-k per vocab shard, an all-gather
    of only K values/ids per shard (O(tp*K), not O(vocab)), then a global top-k. Columns at or
    beyond ``vocab_size`` (LM-head padding) are excluded. Returns (ids int32 [N, K], values
    float32 [N, K]), values sorted descending.
    """
    tp = int(mesh.shape.get("tensor", 1))

    def local(x):
        v_local = x.shape[-1]
        offset = jax.lax.axis_index("tensor") * v_local if tp > 1 else 0
        cols = offset + jnp.arange(v_local, dtype=jnp.int32)
        x = jnp.where(cols[None, :] < vocab_size, x.astype(jnp.float32), -jnp.inf)
        vals, ids = jax.lax.top_k(x, k)
        ids = ids.astype(jnp.int32) + offset
        if tp > 1:
            vals = jax.lax.all_gather(vals, "tensor", axis=1, tiled=True)
            ids = jax.lax.all_gather(ids, "tensor", axis=1, tiled=True)
            vals, sel = jax.lax.top_k(vals, k)
            ids = jnp.take_along_axis(ids, sel, axis=1)
        return ids, vals

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=P("data", "tensor"),
        out_specs=(P("data", None), P("data", None)),
        # After the tiled all-gather + global top-k every tensor shard holds the same result.
        check_vma=False,
    )(logits)


class CandidateSelector(nnx.Module):
    """DFlash2 candidate selector: scores the K x K transitions between adjacent proposal slots.

    Port of upstream ``CandidateSelector``. For proposal slot ``e`` with top-K candidates
    ``cand[b, e, :]`` (from the target LM head) and predecessor ``pred[b, e, p]`` (the verified
    anchor token for slot 0, else ``cand[b, e - 1, p]``):

        score[b, e, p, c] = unary[b, e, c]
                          + <A[pred[b, e, p]] * W h[b, e], B[cand[b, e, c]]>

    with ``A``/``B`` the [vocab, rank] predecessor/successor codebooks and ``W`` the
    hidden projection. The [vocab, rank] tables are replicated (any rank may need any row).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        state_rank: int,
        top_k: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.top_k = int(top_k)
        self.state_rank = int(state_rank)
        self.vocab_size = int(vocab_size)
        self.mesh = mesh
        table = NamedSharding(mesh, P(None, None))
        self.predecessor_codebook = nnx.Param(
            jnp.zeros((self.vocab_size, self.state_rank), dtype=dtype, out_sharding=table)
        )
        self.successor_codebook = nnx.Param(
            jnp.zeros((self.vocab_size, self.state_rank), dtype=dtype, out_sharding=table)
        )
        self.hidden_projection = LinearBase(
            input_size=int(hidden_size),
            output_size=self.state_rank,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="hidden_projection",
        )

    def build_lattice(
        self,
        *,
        candidate_ids: jax.Array,  # [B, E, K] int
        unary_logits: jax.Array,  # [B, E, K] f32
        hidden_states: jax.Array,  # [B, E, H]
        anchor_token_ids: jax.Array,  # [B] int
    ) -> jax.Array:
        """-> scores [B, E, K(pred), K(cand)] float32."""
        b, e, k = candidate_ids.shape
        hidden, _ = self.hidden_projection(
            hidden_states, out_sharding=NamedSharding(self.mesh, P("data", None, None))
        )
        candidate_ids = jax.sharding.reshard(
            candidate_ids.astype(jnp.int32), NamedSharding(self.mesh, P("data", None, None))
        )
        anchor = jax.sharding.reshard(
            anchor_token_ids.astype(jnp.int32), NamedSharding(self.mesh, P("data"))
        )
        anchor = jnp.broadcast_to(anchor[:, None, None], (b, 1, k))
        predecessor_ids = jnp.concatenate([anchor, candidate_ids[:, :-1]], axis=1)
        rows = NamedSharding(self.mesh, P("data", None, None, None))
        keys = self.successor_codebook.value.at[candidate_ids].get(out_sharding=rows)
        predecessors = self.predecessor_codebook.value.at[predecessor_ids].get(out_sharding=rows)
        pair = jnp.einsum(
            "blpr,blcr->blpc",
            predecessors * hidden[:, :, None, :].astype(predecessors.dtype),
            keys,
            preferred_element_type=jnp.float32,
            out_sharding=rows,
        )
        unary = jax.sharding.reshard(
            unary_logits.astype(jnp.float32), NamedSharding(self.mesh, P("data", None, None))
        )
        return unary[:, :, None, :] + pair

    @staticmethod
    def greedy_path(candidate_ids: jax.Array, scores: jax.Array) -> jax.Array:
        """Greedy walk of the lattice (upstream ``sample_path`` with greedy_mask=True).

        Slot 0 takes argmax over its anchor row; each later slot takes the argmax of the row
        of the candidate chosen at the previous slot. Returns tokens [B, E] int32.
        """
        num_slots = scores.shape[1]
        index = jnp.argmax(scores[:, 0, 0], axis=-1).astype(jnp.int32)
        path = [index]
        for slot in range(1, num_slots):
            row = jnp.take_along_axis(scores[:, slot], index[:, None, None], axis=1)[:, 0]
            index = jnp.argmax(row, axis=-1).astype(jnp.int32)
            path.append(index)
        path = jnp.stack(path, axis=1)
        return jnp.take_along_axis(candidate_ids, path[..., None], axis=-1)[..., 0].astype(
            jnp.int32
        )


class DFlash2DraftModel(DFlashDraftModel):
    """DFlash2 draft: DFlash backbone + grouped dynamic convs + low-rank candidate selector.

    The draft has no LM head of its own: candidates come from the target head (passed in by
    the worker), and the selector picks one path through the per-slot top-K candidates.
    """

    uses_candidate_selector = True

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__(config=config, mesh=mesh, dtype=dtype)
        if not self.options.has_selector:
            raise ValueError("DFlash2DraftModel requires dflash_config.selector_rank/top_k.")
        self.block_size = self.options.block_size
        self.candidate_selector = CandidateSelector(
            hidden_size=int(config.hidden_size),
            vocab_size=int(config.vocab_size),
            state_rank=self.options.selector_rank,
            top_k=self.options.selector_top_k,
            mesh=mesh,
            dtype=dtype,
        )

    @staticmethod
    def _model_options(config: PretrainedConfig) -> DFlashModelOptions:
        return get_dflash_model_options(config)

    def set_block_size(self, block_size: int) -> None:
        """Adopt the worker's verify width (upstream ``set_block_size``).

        The grouped convolutions index tokens by in-block position, so the resolved width
        must reach them; the worker calls this before ``nnx.split`` captures the graph.
        """
        self.block_size = int(block_size)
        for layer in self.model.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                if conv is not None:
                    conv.block_size = self.block_size

    def transform_unary_logits(self, logits: jax.Array) -> jax.Array:
        logits = logits.astype(jnp.float32)
        if self.options.output_multiplier != 1.0:
            logits = logits * self.options.output_multiplier
        softcap = self.options.final_logit_softcapping
        if softcap is not None:
            logits = jnp.tanh(logits / softcap) * softcap
        return logits

    def compute_candidates(
        self, hidden: jax.Array, lm_head: jax.Array, vocab_size: int
    ) -> tuple[jax.Array, jax.Array]:
        """hidden [N, H], target lm_head [V_pad, H] -> candidate ids / unary logits [N, K]."""
        logits = jnp.dot(
            hidden,
            lm_head.T,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor")),
        )
        ids, vals = candidate_topk(
            logits, self.candidate_selector.top_k, vocab_size=vocab_size, mesh=self.mesh
        )
        return ids, self.transform_unary_logits(vals)

    def propose_block_tokens(
        self,
        proposal_hidden: jax.Array,  # [B, E, H], draft hiddens of the mask slots
        lm_head: jax.Array,  # [V_pad, H] target head
        anchor_tokens: jax.Array,  # [B] verified token at block position 0
        vocab_size: int,
        return_lattice: bool = False,
    ):
        """Greedy DFlash2 proposal: top-K candidates per slot, lattice, greedy walk."""
        b, e, h = proposal_hidden.shape
        flat = proposal_hidden.reshape(
            b * e, h, out_sharding=NamedSharding(self.mesh, P("data", None))
        )
        ids, unary = self.compute_candidates(flat, lm_head, vocab_size)
        k = ids.shape[-1]
        lattice_sharding = NamedSharding(self.mesh, P("data", None, None))
        ids = ids.reshape(b, e, k, out_sharding=lattice_sharding)
        unary = unary.reshape(b, e, k, out_sharding=lattice_sharding)
        scores = self.candidate_selector.build_lattice(
            candidate_ids=ids,
            unary_logits=unary,
            hidden_states=proposal_hidden,
            anchor_token_ids=anchor_tokens,
        )
        tokens = CandidateSelector.greedy_path(ids, scores)
        if return_lattice:
            return tokens, ids, unary, scores
        return tokens

    def load_weights(self, model_config: ModelConfig) -> None:
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        mappings = self._create_weight_mappings()
        self._check_checkpoint_coverage(set(loader.metadata), mappings)
        loader.load(mappings)
        logger.info("DFlash2 draft weights loaded successfully (%d tensors).", len(mappings))

    def _check_checkpoint_coverage(self, checkpoint_keys: set[str], mappings) -> None:
        """Fail loudly on any checkpoint tensor without a mapping, and vice versa.

        Layers at or beyond ``num_hidden_layers`` are the only tolerated extras (an explicit
        layer truncation of the config, as the generic loader allows for ``model.layers.*``).
        """
        num_layers = int(self.config.num_hidden_layers)

        def truncated(key: str) -> bool:
            parts = key.removeprefix("model.").split(".")
            return parts[0] == "layers" and parts[1].isdigit() and int(parts[1]) >= num_layers

        unmapped = sorted(k for k in checkpoint_keys - set(mappings) if not truncated(k))
        missing = sorted(set(mappings) - checkpoint_keys)
        if unmapped or missing:
            raise ValueError(
                "DFlash2 checkpoint/model mismatch: "
                f"unmapped checkpoint tensors={unmapped[:20]} (total {len(unmapped)}), "
                f"missing checkpoint tensors={missing[:20]} (total {len(missing)})."
            )

    def _create_weight_mappings(self) -> dict[str, WeightSpec]:
        mappings = super()._create_weight_mappings()
        if self.options.has_conv:
            for layer_idx in range(int(self.config.num_hidden_layers)):
                for conv in ("attention_conv", "mlp_conv"):
                    prefix = f"layers.{layer_idx}.{conv}"
                    target = f"model.layers.{layer_idx}.{conv}"
                    mappings[f"{prefix}.base_kernel"] = WeightSpec(
                        target_path=f"{target}.base_kernel",
                        sharding=(None, None, None),
                        transpose=False,
                    )
                    mappings[f"{prefix}.kernel_projection.weight"] = WeightSpec(
                        target_path=f"{target}.kernel_projection.weight",
                        sharding=(None, None),
                        transpose=True,
                    )
        mappings.update(
            {
                "candidate_selector.predecessor_codebook": WeightSpec(
                    target_path="candidate_selector.predecessor_codebook",
                    sharding=(None, None),
                    transpose=False,
                ),
                "candidate_selector.successor_codebook": WeightSpec(
                    target_path="candidate_selector.successor_codebook",
                    sharding=(None, None),
                    transpose=False,
                ),
                "candidate_selector.hidden_projection.weight": WeightSpec(
                    target_path="candidate_selector.hidden_projection.weight",
                    sharding=(None, None),
                    transpose=True,
                ),
            }
        )
        return mappings


class MiniMaxDFlash2DraftModel(DFlash2DraftModel):
    """Compatibility alias for MiniMax DFlash2 checkpoint configs (as upstream)."""


EntryClass = [DFlashDraftModel, DFlash2DraftModel, MiniMaxDFlash2DraftModel]
