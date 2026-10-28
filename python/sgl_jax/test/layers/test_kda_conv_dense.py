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


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_dense_rows_nan_in_unrelated_slot_does_not_leak(dtype):
    rng = np.random.default_rng(1)
    S, B, C, W = 33, 6, 64, 3
    buf = np.asarray(rng.standard_normal((S, C, W)), np.float32)
    buf[7, 3, 1] = np.nan  # unrelated slot (padding slot 0 and the batch's slots stay finite)
    buf[9, 0, 0] = np.inf
    buf = jnp.asarray(buf, dtype)
    idx = jnp.asarray([1, 2, 5, 11, 0, 0], jnp.int32)
    got = np.asarray(_dense_rows_get(buf, idx).astype(jnp.float32))
    np.testing.assert_array_equal(got, np.asarray(buf[idx].astype(jnp.float32)))
    assert np.isfinite(got).all()
    # a selected non-finite slot still reads back non-finite (NaN marker) at that element only
    got7 = np.asarray(_dense_rows_get(buf, jnp.asarray([7, 1], jnp.int32)).astype(jnp.float32))
    assert np.isnan(got7[0, 3, 1]) and np.isfinite(np.delete(got7.reshape(-1), 3 * W + 1)).all()
    # NaN in a value row that is NOT written (write=False) must not touch any slot
    val = np.asarray(rng.standard_normal((B, C, W)), np.float32)
    val[3, 2, 2] = np.nan
    val = jnp.asarray(val, dtype)
    write = jnp.asarray([True, True, True, False, False, False])
    out = np.asarray(_dense_rows_set(buf, idx, val, write).astype(jnp.float32))
    ref = np.asarray(buf.at[idx[:3]].set(val[:3]).astype(jnp.float32))
    np.testing.assert_array_equal(out, ref)
