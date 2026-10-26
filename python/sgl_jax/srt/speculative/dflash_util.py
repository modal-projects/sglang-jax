from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_DFLASH_MASK_TOKEN = "<|MASK|>"
DEFAULT_DFLASH_BLOCK_SIZE = 16


@dataclass(frozen=True)
class DFlashDraftConfig:
    """DFlash settings consumed outside the draft model implementation."""

    block_size: int
    target_layer_ids: list[int] | None
    mask_token: str
    mask_token_id: int | None


def _get_dflash_subconfig(hf_config) -> dict:
    """Return ``dflash_config`` as a dict."""
    cfg = getattr(hf_config, "dflash_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    to_dict = getattr(cfg, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    try:
        return dict(cfg.__dict__)
    except Exception:
        return {}


def _cfg_get(dflash: dict, hf_config, key: str, default=None):
    """Prefer dflash_config[key], fall back to a top-level hf_config attribute."""
    if key in dflash and dflash[key] is not None:
        return dflash[key]
    return getattr(hf_config, key, default)


def parse_dflash_draft_config(
    model_path: str,
    revision: str | None = None,
    trust_remote_code: bool = True,
) -> DFlashDraftConfig:
    """Load a DFlash draft model's HF config."""
    from sgl_jax.srt.hf_transformers_utils import get_config

    hf_config = get_config(model_path, trust_remote_code=trust_remote_code, revision=revision)
    dflash = _get_dflash_subconfig(hf_config)

    block_size = _cfg_get(dflash, hf_config, "block_size", None)
    if block_size is None:
        logger.warning(
            "DFLASH: draft config has no block_size; defaulting to %d.",
            DEFAULT_DFLASH_BLOCK_SIZE,
        )
        block_size = DEFAULT_DFLASH_BLOCK_SIZE
    block_size = int(block_size)
    if block_size <= 1:
        raise ValueError(f"DFLASH block_size must be > 1, got {block_size}.")

    raw_layer_ids = _cfg_get(dflash, hf_config, "target_layer_ids", None)
    if raw_layer_ids is not None:
        target_layer_ids = [int(x) for x in raw_layer_ids]
        if not target_layer_ids:
            raise ValueError("DFLASH target_layer_ids must be non-empty when set.")
    else:
        target_layer_ids = None
        logger.warning(
            "DFLASH: draft config has no target_layer_ids; capture layers will "
            "fall back to defaults, which may mismatch the draft fc in-dim."
        )

    mask_token = _cfg_get(dflash, hf_config, "mask_token", DEFAULT_DFLASH_MASK_TOKEN)
    mask_token_id = _cfg_get(dflash, hf_config, "mask_token_id", None)
    if mask_token_id is not None:
        mask_token_id = int(mask_token_id)

    return DFlashDraftConfig(
        block_size=block_size,
        target_layer_ids=target_layer_ids,
        mask_token=str(mask_token),
        mask_token_id=mask_token_id,
    )


def resolve_mask_token_id(
    dflash_config: DFlashDraftConfig,
    tokenizer,
    vocab_size: int,
) -> int:
    """Resolve and validate the draft mask token id."""
    resolved = dflash_config.mask_token_id
    if resolved is None:
        if tokenizer is None:
            raise RuntimeError(
                "DFLASH requires either dflash_config.mask_token_id or an "
                "initialized tokenizer to resolve mask_token "
                f"{dflash_config.mask_token!r}."
            )
        vocab = tokenizer.get_vocab()
        resolved = vocab.get(dflash_config.mask_token, None)
        if resolved is None:
            raise ValueError(
                f"DFLASH mask_token {dflash_config.mask_token!r} not found in "
                "tokenizer vocab and no mask_token_id given."
            )
    resolved = int(resolved)
    if resolved < 0 or resolved >= int(vocab_size):
        raise ValueError(
            f"DFLASH mask_token_id={resolved} outside target vocab_size={vocab_size}. "
            "Vocab resizing is not supported in minimal mode."
        )
    return resolved


@dataclass(frozen=True)
class DFlashModelOptions:
    """Draft-architecture options read from ``dflash_config`` (DFlash v1 and DFlash2).

    Mirrors the fields of upstream SGLang's ``parse_dflash_draft_config`` that change the
    draft forward: the grouped dynamic convolution (``conv_*``), the candidate selector
    (``selector_*``) and the unary-logit transform (``output_multiplier`` / softcap).
    """

    block_size: int
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int
    output_multiplier: float
    final_logit_softcapping: float | None
    mask_token_id: int | None

    @property
    def has_conv(self) -> bool:
        return self.conv_kernel_size > 0

    @property
    def has_selector(self) -> bool:
        return self.selector_rank > 0


def _nonneg_int(dflash: dict, key: str) -> int:
    value = dflash.get(key, 0)
    if value is None:
        return 0
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"DFLASH {key} must be a non-negative int, got {value!r}.")
    return int(value)


def get_dflash_model_options(hf_config) -> DFlashModelOptions:
    """Parse and validate the draft-forward options of a DFlash/DFlash2 HF config."""
    dflash = _get_dflash_subconfig(hf_config)
    block_size = int(_cfg_get(dflash, hf_config, "block_size", DEFAULT_DFLASH_BLOCK_SIZE))
    conv_kernel_size = _nonneg_int(dflash, "conv_kernel_size")
    conv_group_size = _nonneg_int(dflash, "conv_group_size")
    if bool(conv_kernel_size) != bool(conv_group_size):
        raise ValueError(
            "DFLASH grouped convolution needs conv_kernel_size and conv_group_size together, "
            f"got conv_kernel_size={conv_kernel_size}, conv_group_size={conv_group_size}."
        )
    selector_rank = _nonneg_int(dflash, "selector_rank")
    selector_top_k = _nonneg_int(dflash, "selector_top_k")
    if bool(selector_rank) != bool(selector_top_k):
        raise ValueError(
            "DFLASH selector needs selector_rank and selector_top_k together, "
            f"got rank={selector_rank}, top_k={selector_top_k}."
        )
    output_multiplier = float(dflash.get("output_multiplier", 1.0) or 1.0)
    if output_multiplier <= 0:
        raise ValueError("DFLASH output_multiplier must be positive.")
    softcap = float(dflash.get("final_logit_softcapping") or 0.0)
    # Upstream features this port does not implement must fail loudly, not silently drift.
    for key in ("attention_sink_bias", "attention_value_scale", "projector_type"):
        if dflash.get(key) not in (None, False, "", "linear"):
            raise NotImplementedError(f"DFLASH dflash_config.{key}={dflash[key]!r} is not ported.")
    mask_token_id = _cfg_get(dflash, hf_config, "mask_token_id", None)
    return DFlashModelOptions(
        block_size=block_size,
        conv_kernel_size=conv_kernel_size,
        conv_group_size=conv_group_size,
        selector_rank=selector_rank,
        selector_top_k=selector_top_k,
        output_multiplier=output_multiplier,
        final_logit_softcapping=softcap if softcap > 0 else None,
        mask_token_id=None if mask_token_id is None else int(mask_token_id),
    )
