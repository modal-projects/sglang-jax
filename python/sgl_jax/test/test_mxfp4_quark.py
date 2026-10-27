"""AMD Quark MXFP4 checkpoints (amd/GLM-5.3-Flash-Quark-MXFP4): format decoding, packed-FP4 GMM.

Real-tensor tests read checkpoint data from env paths and skip without them:
  MXFP4_SHARD=<one model-*.safetensors shard of the Quark checkpoint>
  MXFP4_MINI=<a local mini checkpoint dir holding layer-1 routed experts (see test_glm53_mxfp4_load)>
The GMM tests run gmm_v2's packed-FP4 path in Pallas TPU interpret mode on CPU.
"""

import importlib
import os

import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.utils.quantization import mxfp4
from sgl_jax.srt.utils.quantization.quark import (
    BF16,
    FP8_BLOCK128,
    MXFP4,
    QuarkScheme,
    hf_module_to_model_path,
    normalize_module_path,
)

SHARD = os.environ.get("MXFP4_SHARD")
MINI = os.environ.get("MXFP4_MINI")


def _reference_decode(packed: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Independent MXFP4 decode from the OCP MX spec (no lookup table): E2M1 = sign bit 3, exponent
    bits 2:1 (bias 1), mantissa bit 0, subnormal at exponent 0; E8M0 = 2**(code - 127). Elements
    are packed two per byte along K, element 2i in the low nibble."""
    packed = packed.astype(np.int64)
    nib = np.stack([packed & 0xF, packed >> 4], axis=-1).reshape(packed.shape[0], -1)
    sign = np.where(nib & 0x8, -1.0, 1.0)
    e = (nib >> 1) & 0x3
    m = nib & 0x1
    mag = np.where(e == 0, 0.5 * m, (1.0 + 0.5 * m) * np.exp2(e - 1.0))
    scale = np.exp2(codes.astype(np.float64) - 127.0)
    return sign * mag * np.repeat(scale, 32, axis=1)


def _random_pair(rng, n, k, lo=100, hi=140):
    return (
        rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8),
        rng.integers(lo, hi, size=(n, k // 32), dtype=np.uint8),
    )


# ---------------------------------------------------------------- format


def test_e2m1_table_matches_spec():
    codes = np.arange(16, dtype=np.uint8)
    packed = np.tile((codes[0::2] | (codes[1::2] << 4)).reshape(1, 8), (1, 2))  # K = 32
    one = np.full((1, 1), 127, np.uint8)  # 2**0
    want = _reference_decode(packed, one)[0]
    np.testing.assert_array_equal(np.tile(mxfp4.FP4_E2M1_VALUES, 2), want)
    np.testing.assert_array_equal(mxfp4.dequantize_mxfp4(packed, one)[0], want)


def test_random_dequant_bit_exact_and_bf16_exact():
    rng = np.random.default_rng(0)
    packed, codes = _random_pair(rng, 64, 512, lo=1, hi=253)
    want = _reference_decode(packed, codes)
    np.testing.assert_array_equal(mxfp4.dequantize_mxfp4(packed, codes, np.float64), want)
    # bf16 holds every E8M0 code 1..252 times E2M1 (<= 2 significant bits) exactly
    bf = mxfp4.dequantize_mxfp4(packed, codes, ml_dtypes.bfloat16)
    np.testing.assert_array_equal(bf.astype(np.float64), want)


def test_reserved_scale_rejected():
    with pytest.raises(ValueError, match="255"):
        mxfp4.dequantize_mxfp4(np.zeros((2, 16), np.uint8), np.full((2, 1), 255, np.uint8))


def test_packed_layout_roundtrip_and_kernel_constants():
    gmm_v2 = importlib.import_module("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2")
    assert mxfp4.FP4_PACK_CHUNK == gmm_v2.FP4_PACK_CHUNK
    assert mxfp4.FP4_PACKED_SCALE_EXP == gmm_v2.FP4_PACKED_SCALE_EXP
    rng = np.random.default_rng(1)
    packed, codes = _random_pair(rng, 48, 256)
    words, scale = mxfp4.mxfp4_pair_to_packed_fp4(packed, codes)
    assert words.dtype == np.uint32 and words.shape == (256 // 8, 48)
    assert scale.shape == (256 // 32, 48)
    np.testing.assert_array_equal(
        mxfp4.packed_fp4_to_dense(words, scale), _reference_decode(packed, codes).T
    )
    _, scale_bf16 = mxfp4.mxfp4_pair_to_packed_fp4(packed, codes, scale_dtype=ml_dtypes.bfloat16)
    np.testing.assert_array_equal(scale_bf16.astype(np.float32), scale)


def test_kernel_expand_matches_layout():
    """gmm_v2.expand_fp4_packed_e5m2 (the in-kernel SWAR decode), run as plain JAX on CPU."""
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    gmm_v2 = importlib.import_module("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2")
    rng = np.random.default_rng(2)
    packed, codes = _random_pair(rng, 128, 128)
    words, scale = mxfp4.mxfp4_pair_to_packed_fp4(packed, codes)
    orig = pltpu.bitcast
    try:  # pltpu.bitcast is a Mosaic primitive; on CPU use lax's equivalent (same bits)
        gmm_v2.pltpu.bitcast = lambda x, dt: _lax_bitcast(x, dt)
        e5m2 = np.asarray(gmm_v2.expand_fp4_packed_e5m2(jnp.asarray(words)).astype(jnp.float32))
    finally:
        gmm_v2.pltpu.bitcast = orig
    got = e5m2 * np.repeat(scale, 32, axis=0)
    np.testing.assert_array_equal(got, _reference_decode(packed, codes).T)


def _lax_bitcast(x, dtype):
    import jax
    import jax.numpy as jnp

    # uint32 [r, c] -> 4 x 8-bit [4r, c]: byte b of row i becomes row 4i + b (Mosaic sublane order)
    b = jax.lax.bitcast_convert_type(x, jnp.uint8)  # [r, c, 4]
    b = jnp.moveaxis(b, -1, 1).reshape(x.shape[0] * 4, x.shape[1])
    return jax.lax.bitcast_convert_type(b, dtype)


def test_quark_scheme_formats():
    mx = dict(dtype="fp4", qscheme="per_group", group_size=32, scale_format="e8m0", is_dynamic=False)
    fp8 = dict(dtype="fp8_e4m3", qscheme="per_block", block_size=[128, 128], is_dynamic=False)
    p = "model.language_model.layers"
    s = QuarkScheme.from_hf(
        {
            "quant_method": "quark",
            "global_quant_config": {"weight": mx},
            "layer_quant_config": {f"{p}.3.self_attn.q_a_proj": {"weight": fp8}},
            "exclude": [f"{p}.3.self_attn.kv_b_proj", "lm_head"],
            "export": {"weight_format": "real_quantized"},
        }
    )
    assert s.module_format(f"{p}.3.mlp.experts.7.up_proj") == MXFP4
    assert s.module_format(f"{p}.3.self_attn.q_a_proj") == FP8_BLOCK128
    assert s.module_format(f"{p}.3.self_attn.kv_b_proj") == BF16
    assert hf_module_to_model_path(f"{p}.3.self_attn.o_proj") == "model.layers.3.self_attn.o_proj"
    assert normalize_module_path("model/layers[3]/self_attn/o_proj") == "model.layers.3.self_attn.o_proj"
    with pytest.raises(NotImplementedError):
        QuarkScheme.from_hf(
            {"quant_method": "quark", "global_quant_config": {"weight": dict(mx, group_size=16)}}
        )


# ---------------------------------------------------------------- real checkpoint tensors


@pytest.mark.skipif(not SHARD, reason="set MXFP4_SHARD to a Quark checkpoint shard")
def test_real_shard_every_mxfp4_tensor_decodes_exactly():
    from safetensors import safe_open

    n_pairs = 0
    with safe_open(SHARD, framework="np") as f:
        keys = set(f.keys())
        for k in sorted(keys):
            if not k.endswith(".weight_scale") or f.get_slice(k).get_dtype() != "U8":
                continue
            wk = k[: -len("_scale")]
            packed, codes = f.get_tensor(wk), f.get_tensor(k)
            assert packed.dtype == np.uint8 and codes.dtype == np.uint8
            assert codes.shape == (packed.shape[0], packed.shape[1] * 2 // 32), (k, codes.shape)
            assert not np.any(codes == 255), k
            want = _reference_decode(packed, codes)
            got = mxfp4.dequantize_mxfp4(packed, codes, ml_dtypes.bfloat16, key=wk)
            np.testing.assert_array_equal(got.astype(np.float64), want, err_msg=wk)
            if n_pairs % 50 == 0:  # packed layout on a subset (slow in numpy)
                words, scale = mxfp4.mxfp4_pair_to_packed_fp4(packed, codes, key=wk)
                np.testing.assert_array_equal(mxfp4.packed_fp4_to_dense(words, scale), want.T)
            n_pairs += 1
    assert n_pairs > 100, n_pairs


# ---------------------------------------------------------------- packed-FP4 GMM (interpret)


@pytest.fixture
def cpu_v6e():
    """Make the gmm_v2 tiler see a v6e chip on CPU."""
    from jax.experimental.pallas import tpu as pltpu

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
    saved = []
    mods = ["jax._src.tpu_info", "jax._src.pallas.mosaic.tpu_info"]
    for name in mods:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        saved.append((mod, mod.get_tpu_info))
        mod.get_tpu_info = lambda: info
    saved.append((pltpu, pltpu.get_tpu_info))
    pltpu.get_tpu_info = lambda: info
    tts = importlib.import_module("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.tuned_tile_sizes")
    saved_dn = tts.get_device_name
    tts.get_device_name = lambda num_devices=None: "TPU v6e"
    yield info
    for mod, fn in saved:
        mod.get_tpu_info = fn
    tts.get_device_name = saved_dn


def _expert_pairs(n_experts, proj):
    if MINI:
        from safetensors import safe_open

        path = os.path.join(MINI, "model-00001-of-00001.safetensors")
        with safe_open(path, framework="np") as f:
            p = "model.language_model.layers.1.mlp.experts"
            return [
                (f.get_tensor(f"{p}.{e}.{proj}.weight"), f.get_tensor(f"{p}.{e}.{proj}.weight_scale"))
                for e in range(n_experts)
            ], True
    rng = np.random.default_rng(5)
    n, k = (4096, 2048) if proj == "down_proj" else (2048, 4096)
    return [_random_pair(rng, n, k, 118, 124) for _ in range(n_experts)], False


@pytest.mark.parametrize("proj", ["gate_proj", "down_proj"])
@pytest.mark.parametrize("rows_per_expert", [1, 8, 128])
def test_gmm_v2_fp4_packed_matches_dequantized_reference(cpu_v6e, proj, rows_per_expert):
    import jax.numpy as jnp

    from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm

    groups = 2 if rows_per_expert >= 128 else 3
    pairs, real = _expert_pairs(groups, proj)
    ws, ss, refs = [], [], []
    for packed, codes in pairs:
        w, s = mxfp4.mxfp4_pair_to_packed_fp4(packed, codes, scale_dtype=ml_dtypes.bfloat16)
        ws.append(w)
        ss.append(s)
        refs.append(mxfp4.dequantize_mxfp4(packed, codes, np.float32).T)  # [K, N]
    k = refs[0].shape[0]
    m = groups * rows_per_expert
    rng = np.random.default_rng(7)
    lhs = jnp.asarray(rng.standard_normal((m, k)).astype(np.float32), jnp.bfloat16)
    out = gmm(
        lhs=lhs,
        rhs=jnp.asarray(np.stack(ws)),
        group_sizes=jnp.full((groups,), rows_per_expert, jnp.int32),
        rhs_scale=jnp.asarray(np.stack(ss)),
        preferred_element_type=jnp.float32,
        maybe_quantize_lhs=False,
        rhs_fp4_packed=True,
        interpret=True,
    )
    lhs32 = np.asarray(lhs, np.float64)
    want = np.concatenate(
        [lhs32[g * rows_per_expert : (g + 1) * rows_per_expert] @ refs[g] for g in range(groups)]
    )
    got = np.asarray(out, np.float64)
    err = np.abs(got - want)
    rel = err.max() / np.abs(want).max()
    print(
        f"\nGMM_PARITY proj={proj} real={real} experts={groups} rows/expert={rows_per_expert} "
        f"K={k} N={refs[0].shape[1]} max_abs={err.max():.3e} max_rel(of max)={rel:.3e} "
        f"ref_max={np.abs(want).max():.3e}"
    )
    # Exact weights and exact bf16 activations: only f32 accumulation order differs.
    assert rel < 1e-5, rel


def test_fp4_path_selection_knobs(monkeypatch):
    gmm_v2 = importlib.import_module("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2")
    # EP8 at M=1024 tokens: 8192 rows over all 288 experts, 36 local groups
    d = gmm_v2.Dimensions(size_m=8192, size_k=4096, size_n=2048, size_group=36,
                          size_lhs_group=288, size_lhs_sublane=16)
    for k in ("SGLJAX_FP4_PATH", "SGLJAX_FP4_FULLK_MIN_ROWS", "SGLJAX_FP4_FULLK_BASIS"):
        monkeypatch.delenv(k, raising=False)
    assert gmm_v2.fp4_packed_use_full_k(d)  # legacy: 8192 >= 96 * 36
    monkeypatch.setenv("SGLJAX_FP4_FULLK_BASIS", "global")
    assert not gmm_v2.fp4_packed_use_full_k(d)  # 8192 / 288 = 28 rows/expert < 96
    monkeypatch.setenv("SGLJAX_FP4_FULLK_MIN_ROWS", "16")
    assert gmm_v2.fp4_packed_use_full_k(d)
    monkeypatch.setenv("SGLJAX_FP4_PATH", "decode")
    assert not gmm_v2.fp4_packed_use_full_k(d)
    monkeypatch.setenv("SGLJAX_FP4_PATH", "fullk")
    assert gmm_v2.fp4_packed_use_full_k(d)
