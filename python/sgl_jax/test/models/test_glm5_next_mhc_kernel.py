"""Glm5NextHC with the Pallas mHC kernels (SGLJAX_GLM5_MHC_KERNEL=1) matches the XLA reference.
CPU: XLA_FLAGS=--xla_force_host_platform_device_count=8 PALLAS_INTERPRET=1 JAX_PLATFORMS=cpu pytest <this file>"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import sgl_jax.srt.kernels.mhc.mhc as _mhc
from sgl_jax.srt.models.glm5_next import Glm5NextHC


@pytest.mark.skipif(
    jax.devices()[0].platform == "cpu" and os.environ.get("PALLAS_INTERPRET") != "1",
    reason="needs TPU or PALLAS_INTERPRET=1",
)
@pytest.mark.parametrize("dp", [1, 4])
def test_glm5_next_mhc_kernel_matches_reference(monkeypatch, dp):
    if jax.devices()[0].platform == "cpu":
        monkeypatch.setattr(_mhc, "_device_kind", lambda: "TPU v6e")
    if len(jax.devices()) % dp:
        pytest.skip("needs a device count divisible by dp")
    mesh = Mesh(
        np.array(jax.devices()).reshape(dp, -1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    D, HC = 4096, 4
    with jax.set_mesh(mesh):
        hc = Glm5NextHC(D, HC, 20, 1e-6, 1e-6, mesh, jnp.bfloat16)
        k = jax.random.split(jax.random.PRNGKey(0), 4)
        hc.fn.value = (jax.random.normal(k[0], hc.fn.value.shape) * 0.02).astype(jnp.bfloat16)
        hc.base.value = jax.random.normal(k[1], hc.base.value.shape) * 0.5
        hc.scale.value = jnp.array([0.7, 1.3, 2.0], jnp.float32)
        worst = 0.0
        for T in ((1, 8, 33) if dp == 1 else (4, 32)):  # DP: tokens split over data
            s = jax.random.normal(jax.random.fold_in(k[2], T), (T, HC * D), jnp.float32).astype(jnp.bfloat16)
            y = jax.random.normal(jax.random.fold_in(k[3], T), (T, D), jnp.float32).astype(jnp.bfloat16)
            if dp > 1:  # DP attention: hidden streams arrive sharded on tokens
                s = jax.sharding.reshard(s, P("data", None))
                y = jax.sharding.reshard(y, P("data", None))
            outs = {}
            for flag in (False, True):
                hc.use_kernel = flag
                c, st = jax.jit(lambda s: hc.pre(s))(s)
                o = jax.jit(lambda y, st, s: hc.post(y, st, s))(y, st, s)
                outs[flag] = (c, st[0], st[1], o)
            for name, a, b in zip(("collapsed", "post", "comb", "out"), outs[False], outs[True]):
                a32, b32 = np.asarray(a, np.float32), np.asarray(b, np.float32)
                assert a32.shape == b32.shape, (name, a32.shape, b32.shape)
                rel = float(np.max(np.abs(a32 - b32)) / (np.max(np.abs(a32)) + 1e-9))
                worst = max(worst, rel)
                print(f"T={T:3d} {name:9s} shape={a32.shape} dtype={a.dtype}/{b.dtype} max_rel={rel:.2e}")
        assert worst < 2e-2, worst
    pass
