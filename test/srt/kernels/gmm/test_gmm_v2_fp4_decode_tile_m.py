"""Packed-FP4 gmm_v2 decode path: tile_m 16 / auto equals tile_m 32 and the dense reference
(CPU interpret, v6e tiling).  JAX_PLATFORMS=cpu python -m pytest <this file>"""
import os

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu


def _patch_tpu_info():
    import importlib

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
    pltpu.get_tpu_info = lambda: info
    for mod in ("jax._src.tpu_info", "jax._src.pallas.mosaic.tpu_info"):
        try:
            importlib.import_module(mod).get_tpu_info = lambda: info
        except Exception:  # noqa: BLE001
            pass
    from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel import tuned_tile_sizes as tts

    tts.get_device_name = lambda num_devices=None: "TPU v6e"


def _run(env, x, w, s, gs, off):
    from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel import gmm_v2 as g2

    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        jax.clear_caches()
        out = g2.gmm_v2(x, w, gs, rhs_scale=s, group_offset=off, preferred_element_type=jnp.bfloat16,
                        acc_dtype=jnp.float32, maybe_quantize_lhs=False, rhs_fp4_packed=True, zero_initialize=False,
                        interpret=pltpu.InterpretParams(uninitialized_memory="zero", out_of_bounds_reads="uninitialized"))
        return np.asarray(out.astype(jnp.float32))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.parametrize("sizes", [(5, 0, 20, 40, 3, 12), (1, 2, 3, 4, 5, 1)])
def test_decode_tile_m(sizes):
    _patch_tpu_info()
    from sgl_jax.srt.utils.quantization.mxfp4 import packed_fp4_to_dense

    rng = np.random.default_rng(0)
    K, N, G = 256, 256, 6
    gs = np.asarray(sizes, np.int32)
    m = int(gs.sum())
    w = rng.integers(0, 2**32, size=(G, K // 8, N), dtype=np.uint32)
    e = rng.integers(-4, 5, size=(G, K // 32, N))
    s = np.ldexp(np.ones(e.shape, np.float32), e + 14).astype(ml_dtypes.bfloat16)
    x = (rng.standard_normal((m, K)) * 0.5).astype(ml_dtypes.bfloat16)
    args = (jnp.asarray(x), jnp.asarray(w), jnp.asarray(s), jnp.asarray(gs), jnp.asarray([0], jnp.int32))
    base = {"SGLJAX_FP4_PATH": "decode"}
    o32 = _run({**base, "SGLJAX_FP4_DECODE_TILE_M": "32"}, *args)
    o16 = _run({**base, "SGLJAX_FP4_DECODE_TILE_M": "16"}, *args)
    o8 = _run({**base, "SGLJAX_FP4_DECODE_TILE_M": "8"}, *args)
    oau = _run({**base, "SGLJAX_FP4_DECODE_TILE_M": "auto"}, *args)
    lo, hi = 0, m
    np.testing.assert_array_equal(o16[lo:hi], o32[lo:hi])
    np.testing.assert_array_equal(oau[lo:hi], o32[lo:hi])
    np.testing.assert_array_equal(o8[lo:hi], o32[lo:hi])
    # auto + MoE rows-per-expert hint: 5 -> tile 16, 20 -> tile 32, 50 -> full-K path
    from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel import gmm_v2 as g2

    env = {"SGLJAX_FP4_PATH": "auto", "SGLJAX_FP4_DECODE_TILE_M": "auto"}
    for hint in (5.0, 20.0, 50.0):
        with g2.fp4_rows_per_expert_hint(hint):
            oh = _run(env, *args)
        np.testing.assert_allclose(oh[lo:hi], o32[lo:hi], rtol=2e-2, atol=2e-2)
    ref = np.zeros((m, N), np.float32)
    r = lo
    for g in range(G):
        n_g = int(gs[g])
        dense = packed_fp4_to_dense(w[g], np.asarray(s[g], np.float32))
        ref[r : r + n_g] = np.asarray(x[r : r + n_g], np.float32) @ dense
        r += n_g
    refb = ref.astype(ml_dtypes.bfloat16).astype(np.float32)
    den = np.abs(ref[lo:hi]).max()
    assert np.abs(o16[lo:hi] - refb[lo:hi]).max() / den < 1e-2
