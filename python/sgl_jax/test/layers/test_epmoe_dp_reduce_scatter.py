"""EPMoE under DP attention: the opt-in reduce-scatter combine (SGLJAX_EPMOE_DP_RS) equals the psum combine.
CPU: XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu pytest <this file>"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers import moe as moe_mod
from sgl_jax.srt.layers.moe import EPMoE


@pytest.mark.skipif(len(jax.devices()) < 8, reason="needs 8 (fake) devices")
def test_epmoe_dp_reduce_scatter_matches_psum(monkeypatch):
    mesh = Mesh(np.asarray(jax.devices()[:8]).reshape(4, 2), ("data", "tensor"),
                axis_types=(AxisType.Explicit,) * 2)
    H, E, K, I, T = 256, 16, 2, 128, 32
    with jax.set_mesh(mesh):
        layer = EPMoE(hidden_size=H, num_experts=E, num_experts_per_tok=K, ep_size=8, mesh=mesh,
                      intermediate_dim=I)
        ks = jax.random.split(jax.random.PRNGKey(0), 6)
        for name, k in zip(("wi_0", "wi_1", "wo"), ks[:3]):
            p = getattr(layer, name)
            p.value = jax.device_put(
                (jax.random.normal(k, p.value.shape, jnp.float32) * 0.05).astype(jnp.bfloat16),
                p.value.sharding)
        x = jax.sharding.reshard(jax.random.normal(ks[3], (T, H), jnp.float32).astype(jnp.bfloat16),
                                 P("data", None))
        ids = jax.sharding.reshard(jax.random.randint(ks[4], (T, K), 0, E, jnp.int32), P("data", None))
        w = jax.sharding.reshard(jax.nn.softmax(jax.random.normal(ks[5], (T, K)), -1), P("data", None))
        out_sh = NamedSharding(mesh, P("data", None))
        outs = {}
        for flag in (False, True):
            monkeypatch.setattr(moe_mod, "_EPMOE_DP_RS", flag)
            f = jax.jit(lambda x, w, i: layer(x, w, i, out_sharding=out_sh))
            hlo = f.lower(x, w, ids).as_text()
            assert ("all_to_all" in hlo) == flag, "a2a reduce-scatter combine must appear iff enabled"
            outs[flag] = np.asarray(f(x, w, ids), np.float32)
    assert np.abs(outs[True]).max() > 0
    np.testing.assert_allclose(outs[True], outs[False], rtol=2e-2, atol=2e-2)
