"""AMD Quark (``quant_method: quark``) checkpoint schemes -> per-module storage formats.

Quark exports one ``global_quant_config`` plus per-module ``layer_quant_config`` overrides and an
``exclude`` list (modules kept in the checkpoint dtype). ``amd/GLM-5.3-Flash-Quark-MXFP4`` uses:

  * global: weight ``fp4`` per_group 32, E8M0 scale (MXFP4), dynamic MXFP4 activations (W4A4 on
    MI355X). On TPU the weights stay MXFP4 and activations stay bf16 (W4A16).
  * overrides: ``fp8_e4m3`` per_block [128, 128] with f32 scales (dense MLP of layers 0-2, the DSA
    layers' q_a/q_b/kv_a/o projections, everything in the MTP layer). These tensors are
    bit-identical to the official ``zai-org`` FP8 release (``weight_scale`` == ``weight_scale_inv``).
  * exclude: KDA projections, router gates, indexers, kv_b_proj, lm_head, mHC, vision (bf16).

Only formats that sglang-jax serves exactly are accepted; anything else fails loudly.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import re

MXFP4 = "mxfp4"
FP8_BLOCK128 = "fp8_block128"
BF16 = "bf16"

# HF text-model prefix of multimodal checkpoints (GLM-5.3-Flash) -> model param prefix.
_HF_TEXT_PREFIXES = ("model.language_model.", "language_model.model.")


def _classify(cfg: dict | None, where: str) -> str:
    if not cfg or cfg.get("weight") is None:
        return BF16
    w = cfg["weight"]
    dtype, qscheme = w.get("dtype"), w.get("qscheme")
    if (
        dtype == "fp4"
        and qscheme == "per_group"
        and w.get("group_size") == 32
        and w.get("scale_format") == "e8m0"
        and not w.get("is_dynamic", False)
    ):
        return MXFP4
    if (
        dtype == "fp8_e4m3"
        and qscheme == "per_block"
        and list(w.get("block_size") or []) == [128, 128]
        and not w.get("is_dynamic", False)
    ):
        return FP8_BLOCK128
    raise NotImplementedError(
        f"Quark weight scheme of {where} is not supported on TPU: "
        f"dtype={dtype} qscheme={qscheme} group_size={w.get('group_size')} "
        f"block_size={w.get('block_size')} scale_format={w.get('scale_format')}"
    )


@dataclasses.dataclass(frozen=True)
class QuarkScheme:
    """Storage format per HF module name (``model.language_model.layers.3.mlp.experts.0.up_proj``)."""

    global_format: str
    overrides: dict  # exact HF module name -> format
    exclude: tuple  # HF module names / fnmatch patterns kept unquantized

    @classmethod
    def from_hf(cls, qc: dict) -> "QuarkScheme":
        if qc.get("quant_method") != "quark":
            raise ValueError(f"not a Quark config: quant_method={qc.get('quant_method')!r}")
        if qc.get("layer_type_quant_config"):
            raise NotImplementedError("Quark layer_type_quant_config is not supported")
        export = qc.get("export") or {}
        if export.get("weight_format", "real_quantized") != "real_quantized":
            raise NotImplementedError(f"Quark export weight_format {export.get('weight_format')!r}")
        if qc.get("kv_cache_quant_config"):
            raise NotImplementedError("Quark KV-cache quantization is not supported")
        overrides = {
            name: _classify(cfg, name) for name, cfg in (qc.get("layer_quant_config") or {}).items()
        }
        return cls(
            global_format=_classify(qc.get("global_quant_config"), "global_quant_config"),
            overrides=overrides,
            exclude=tuple(qc.get("exclude") or ()),
        )

    def is_excluded(self, module: str) -> bool:
        return any(module == e or fnmatch.fnmatchcase(module, e) for e in self.exclude)

    def module_format(self, module: str) -> str:
        """Format of HF module ``module`` (no ``.weight`` suffix)."""
        if self.is_excluded(module):
            return BF16
        return self.overrides.get(module, self.global_format)

    def to_dict(self) -> dict:
        return {
            "global_format": self.global_format,
            "num_overrides": len(self.overrides),
            "num_exclude": len(self.exclude),
        }


def hf_module_to_model_path(module: str) -> str:
    """``model.language_model.layers.3.self_attn.o_proj`` -> ``model.layers.3.self_attn.o_proj``."""
    for p in _HF_TEXT_PREFIXES:
        if module.startswith(p):
            return "model." + module[len(p) :]
    return module


def normalize_module_path(path: str) -> str:
    """Model-walker path (``model/layers[3]/self_attn/o_proj``) or dot path -> dot path with
    numeric indices as plain components (``model.layers.3.self_attn.o_proj``)."""
    path = path.replace("/", ".")
    path = re.sub(r"\[(\d+)\]", r".\1", path)
    return re.sub(r"\.+", ".", path).strip(".")


def fp8_linear_module_paths(scheme: QuarkScheme) -> list[str]:
    """Model paths of the non-expert linears the checkpoint stores as FP8 block-128."""
    return sorted(
        hf_module_to_model_path(name)
        for name, fmt in scheme.overrides.items()
        if fmt == FP8_BLOCK128 and ".experts." not in name
    )


def apply_quark_scheme(quant_config, hf_quant_config: dict):
    """Attach a Quark checkpoint's scheme to ``quant_config`` (a QuantizationConfig).

    Linear rules without an explicit module list are restricted to the modules the checkpoint
    stores as FP8 block-128 (``module_paths``); every other linear stays a plain bf16 LinearBase.
    MXFP4 non-expert linears (GLM-5.3 shared experts) are dequantized exactly to bf16 by the model's
    weight mappings, MXFP4 routed experts stay packed (``moe_weight_format: mxfp4``).
    """
    scheme = QuarkScheme.from_hf(hf_quant_config)
    if scheme.global_format != MXFP4:
        raise NotImplementedError(
            f"Quark checkpoints are supported with an MXFP4 global scheme, got {scheme.global_format}"
        )
    fp8_paths = fp8_linear_module_paths(scheme)
    quant_config.quark_scheme = scheme
    quant_config.is_static_checkpoint = True
    if quant_config.moe_weight_format is None:
        import jax.numpy as jnp

        quant_config.moe_weight_format = MXFP4
        quant_config.moe_weight_dtype = jnp.uint32
        quant_config.moe_activation_dtype = None
    if quant_config.moe_weight_format != MXFP4:
        raise ValueError("Quark MXFP4 checkpoints need quantization.moe.weight_format: mxfp4")
    rules = []
    for rule in quant_config.linear_rules or []:
        rule = dict(rule)
        if "module_paths" not in rule:
            rule["module_paths"] = fp8_paths
        rules.append(rule)
    quant_config.linear_rules = rules
    if fp8_paths and quant_config.weight_block_size is None:
        quant_config.weight_block_size = (128, 128)
    if tuple(quant_config.weight_block_size or ()) not in ((), (128, 128)):
        raise ValueError(
            f"Quark FP8 linears are block-128; weight_block_size={quant_config.weight_block_size}"
        )
    quant_config.allow_narrow_n_blockwise = True
    ignored = list(quant_config.ignored_layers or [])
    ignored += [hf_module_to_model_path(e) for e in scheme.exclude if "*" not in e]
    quant_config.ignored_layers = ignored
    return quant_config


def quark_quantization_config(hf_quant_config: dict):
    """QuantizationConfig for a Quark MXFP4 checkpoint auto-detected from config.json."""
    import jax.numpy as jnp

    from sgl_jax.srt.configs.quantization_config import QuantizationConfig

    qc = QuantizationConfig(
        is_static_checkpoint=True,
        linear_rules=[
            {"module_path": ".*", "weight_dtype": "float8_e4m3fn", "activation_dtype": None}
        ],
        moe_weight_dtype=jnp.uint32,
        moe_activation_dtype=None,
        moe_weight_format=MXFP4,
        weight_block_size=(128, 128),
        allow_narrow_n_blockwise=True,
    )
    return apply_quark_scheme(qc, hf_quant_config)
