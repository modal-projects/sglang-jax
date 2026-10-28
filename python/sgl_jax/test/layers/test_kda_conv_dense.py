"""SGLJAX_KDA_CONV_DENSE helpers == XLA row gather / guarded scatter (idx 0 = padding)."""
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.attention.linear.kda_backend import _dense_rows_get, _dense_rows_set


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_dense_rows_get_set_match_scatter(dtype):
    rng = np.random.default_rng(0)
    S, B, C, W = 65, 12, 96, 3
    buf = jnp.asarray(rng.standard_normal((S, C, W)), dtype)
    idx = np.r_[rng.permutation(np.arange(1, S))[: B - 3], [0, 0, 0]].astype(np.int32)  # 3 padding rows
    idx = jnp.asarray(idx)
    val = jnp.asarray(rng.standard_normal((B, C, W)), dtype)
    np.testing.assert_array_equal(np.asarray(_dense_rows_get(buf, idx)), np.asarray(buf[idx]))
    keep = (idx == 0).reshape(-1, 1, 1)
    ref = buf.at[idx].set(jnp.where(keep, buf[idx], val))
    got = _dense_rows_set(buf, idx, val, idx != 0)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
