"""SGLJAX_FP4_PACK=v2 packing: host packer/unpacker round trip, in-kernel E5M2 expansion, and the
packed-FP4 gmm (decode tile 16 and full-K) vs the dense reference (CPU interpret)."""
import importlib.util
import os

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

_T = os.path.join(os.path.dirname(__file__), "test_gmm_v2_fp4_decode_tile_m.py")
_spec = importlib.util.spec_from_file_location("_tm", _T)
_tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tm)


@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setenv("SGLJAX_FP4_PACK", "v2")


def _pair(rng, n, k):
    packed = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    codes = rng.integers(110, 135, size=(n, k // 32), dtype=np.uint8)
    return packed, codes


def test_roundtrip_and_expand(v2):
    from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel import gmm_v2 as g2
    from sgl_jax.srt.utils.quantization import mxfp4

    rng = np.random.default_rng(1)
    packed, codes = _pair(rng, 256, 512)
    want = mxfp4.dequantize_mxfp4(packed, codes).T  # [K, N]
    words, scale = mxfp4.mxfp4_pair_to_packed_fp4(packed, codes)
    np.testing.assert_array_equal(mxfp4.packed_fp4_to_dense(words, scale), want)
    e5 = np.asarray(g2.expand_fp4_packed_e5m2(jnp.asarray(words)).astype(jnp.float32))
    sc = np.repeat(scale, 32, axis=0) * np.float32(2.0**-14)
    np.testing.assert_array_equal(e5 * sc, want)
    # v1 words decode differently (the layouts really differ)
    os.environ["SGLJAX_FP4_PACK"] = "v1"
    assert not np.array_equal(mxfp4.packed_fp4_to_dense(words, scale), want)


@pytest.mark.parametrize("path,tile", [("decode", "16"), ("decode", "32"), ("fullk", "32")])
def test_gmm_v2_pack_v2(v2, path, tile):
    _tm._patch_tpu_info()
    from sgl_jax.srt.utils.quantization import mxfp4

    rng = np.random.default_rng(2)
    K, N, G = 256, 256, 3
    gs = np.asarray([5, 20, 7], np.int32)
    ws, ss, dense = [], [], []
    for _ in range(G):
        packed, codes = _pair(rng, N, K)
        w, s = mxfp4.mxfp4_pair_to_packed_fp4(packed, codes, scale_dtype=ml_dtypes.bfloat16)
        ws.append(w); ss.append(s); dense.append(mxfp4.dequantize_mxfp4(packed, codes).T)
    m = int(gs.sum())
    x = (rng.standard_normal((m, K)) * 0.5).astype(ml_dtypes.bfloat16)
    out = _tm._run({"SGLJAX_FP4_PATH": path, "SGLJAX_FP4_DECODE_TILE_M": tile, "SGLJAX_FP4_PACK": "v2"},
                   jnp.asarray(x), jnp.asarray(np.stack(ws)), jnp.asarray(np.stack(ss)),
                   jnp.asarray(gs), jnp.asarray([0], jnp.int32))
    ref = np.concatenate([np.asarray(x[a:b], np.float32) @ dense[g]
                          for g, (a, b) in enumerate(zip(np.r_[0, np.cumsum(gs)[:-1]], np.cumsum(gs)))])
    refb = ref.astype(ml_dtypes.bfloat16).astype(np.float32)
    assert np.abs(out - refb).max() / np.abs(ref).max() < 1e-2
