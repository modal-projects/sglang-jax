"""MXFP4 experts kept resident as FP4: exact checkpoint values, GMM layout, and GMM numerics."""

import os

import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.layers.deepseek_v4_moe_loader import mxfp4_pair_to_resident_fp4
from sgl_jax.srt.utils.quantization.mxfp4_fp8_loader import FP4_CODEBOOK


def _reference_dequant(packed, codes):
    """Independent decode: codebook lookup per nibble (low first) times 2**(code-127)."""
    n, half_k = packed.shape
    nib = np.empty((n, 2 * half_k), np.uint8)
    nib[:, 0::2] = packed & 0x0F
    nib[:, 1::2] = packed >> 4
    vals = FP4_CODEBOOK[nib]
    scale = np.exp2(codes.astype(np.float64) - 127).astype(np.float32)
    return vals * np.repeat(scale, 32, axis=1)  # [N, K]


def _random_pair(rng, n, k):
    packed = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    codes = rng.integers(100, 140, size=(n, k // 32), dtype=np.uint8)
    return packed, codes


def test_resident_fp4_is_exact_and_in_gmm_layout():
    rng = np.random.default_rng(0)
    n, k = 96, 256
    packed, codes = _random_pair(rng, n, k)
    weight, scale = mxfp4_pair_to_resident_fp4(packed.view(np.int8), codes, key="t")
    assert weight.dtype == ml_dtypes.int4 and weight.shape == (k, n)
    assert scale.dtype == ml_dtypes.bfloat16 and scale.shape == (k // 32, 1, n)
    from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import E2M1_DECODE_SCALE_EXP

    codes4 = weight.astype(np.int8).astype(np.int16) % 16  # back to nibble codes 0..15
    scale32 = np.repeat(scale[:, 0, :].astype(np.float32), 32, axis=0)
    got = FP4_CODEBOOK[codes4] * scale32 * np.float32(2.0**-E2M1_DECODE_SCALE_EXP)
    np.testing.assert_array_equal(got, _reference_dequant(packed, codes).T)


def test_decode_e2m1_codes_matches_codebook():
    import jax.numpy as jnp

    from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import (
        E2M1_DECODE_SCALE_EXP,
        decode_e2m1_codes,
    )

    codes = np.arange(16, dtype=np.uint8)
    v = (codes.astype(np.int8) - 16 * (codes >= 8)).astype(ml_dtypes.int4)
    got = np.asarray(decode_e2m1_codes(jnp.asarray(v), jnp.float32))
    np.testing.assert_array_equal(got * 2.0**E2M1_DECODE_SCALE_EXP, FP4_CODEBOOK)


def test_reserved_scale_code_rejected():
    packed = np.zeros((4, 16), np.uint8)
    codes = np.full((4, 1), 255, np.uint8)
    with pytest.raises(ValueError, match="reserved"):
        mxfp4_pair_to_resident_fp4(packed, codes, key="t")


@pytest.mark.skipif(os.environ.get("JAX_PLATFORMS", "") == "cpu", reason="GMM v2 needs a TPU")
@pytest.mark.parametrize("m_per_group", [1, 7, 64, 300])
def test_gmm_v2_fp4_block_scale_matches_dequantized_reference(m_per_group):
    import jax
    import jax.numpy as jnp

    from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm

    if jax.devices()[0].platform != "tpu":
        pytest.skip("needs TPU")
    rng = np.random.default_rng(1)
    groups, k, n = 4, 4096, 2048  # V4 Flash expert shapes: H=4096, I=2048
    ws, ss, refs = [], [], []
    for _ in range(groups):
        packed, codes = _random_pair(rng, n, k)
        w, s = mxfp4_pair_to_resident_fp4(packed, codes, key="t")
        ws.append(w)
        ss.append(s)
        refs.append(_reference_dequant(packed, codes).T)
    rhs = jnp.asarray(np.stack(ws))
    rhs_scale = jnp.asarray(np.stack(ss))
    group_sizes = jnp.full((groups,), m_per_group, jnp.int32)
    m = groups * m_per_group
    lhs = jnp.asarray(rng.standard_normal((m, k)).astype(np.float32) * 0.05, jnp.bfloat16)

    # Through the backend, which pads rows to the v2 alignment like EPMoE's calls.
    out = gmm(
        lhs=lhs, rhs=rhs, group_sizes=group_sizes, rhs_scale=rhs_scale,
        preferred_element_type=jnp.float32, maybe_quantize_lhs=False, rhs_e2m1_codes=True,
    )[:m]
    out = np.asarray(out, np.float32)

    lhs32 = np.asarray(lhs, np.float32)
    want = np.concatenate(
        [lhs32[g * m_per_group : (g + 1) * m_per_group] @ refs[g] for g in range(groups)]
    )
    # Weights are exact in bf16 after scaling; the only error is bf16 rounding of w*scale
    # products being exact (powers of two) and f32 accumulation order.
    np.testing.assert_allclose(out, want, rtol=2e-2, atol=2e-2 * np.abs(want).max())
