"""KDA TARGET_VERIFY (multi-token window, no pool writes) + commit at the accepted position.

Checked against the existing single-token DECODE path run sequentially over the same tokens:
  * verify outputs at every window position == the t-th sequential decode output;
  * commit_kda_verify(record, n) == pool state after n sequential decode steps, for every
    per-request n in [0, T] (n = seed + accepted drafts in the DFlash worker);
  * padded rows (recurrent index 0) never write the dummy slot.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
)
from sgl_jax.srt.layers.attention.linear.kda_backend import (
    KDAVerifyRecord,
    commit_kda_verify,
    commit_recurrent_verify,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.test.test_kda_attention import (
    create_test_data,
    gather_conv,
    gather_ssm,
    mesh,
)

NUM_HEADS, HEAD_DIM, CONV_K, T = 4, 16, 4, 5


def _put(x, spec):
    return jax.device_put(x, NamedSharding(mesh, spec))


def _setup(batch_size, dtype, seed=0):
    rng = np.random.default_rng(seed)
    fb, pool, layer, *_rest, recurrent_indices = create_test_data(
        "decode",
        [1] * batch_size,
        NUM_HEADS,
        HEAD_DIM,
        CONV_K,
        dtype,
        rng,
        all_have_initial_state=True,
    )
    hidden = NUM_HEADS * HEAD_DIM

    def normal(shape, scale=0.1):
        return jnp.asarray(rng.standard_normal(shape).astype(np.float32) * scale, dtype=dtype)

    tokens = dict(
        q=normal((batch_size, T, hidden)),
        k=normal((batch_size, T, hidden)),
        v=normal((batch_size, T, hidden)),
        a=normal((batch_size, T, hidden)),
        b=jax.nn.sigmoid(normal((batch_size, T, NUM_HEADS), 1.0).astype(jnp.float32)),
    )
    return fb, pool, layer, recurrent_indices, tokens


def _set_meta(fb, mode, batch_size, recurrent_indices, width):
    meta = fb.attn_backend.forward_metadata
    fb.forward_mode = mode
    meta.cu_q_lens = _put(jnp.arange(batch_size + 1, dtype=jnp.int32) * width, P("data"))
    meta.recurrent_indices = _put(jnp.asarray(recurrent_indices, jnp.int32), P("data"))
    meta.has_initial_state = _put(jnp.ones((batch_size,), jnp.bool_), P("data"))


def _call(layer, fb, pool, toks):
    hs = P("data", "tensor")
    return layer(fb, *(_put(toks[n], hs) for n in "qkvab"), pool)


def _snapshot(pool, recurrent_indices):
    ssm_buf, conv_list = pool.get_linear_recurrent_layer_cache(0)
    return (
        np.asarray(gather_ssm(pool, ssm_buf, recurrent_indices)),
        np.asarray(gather_conv(pool, conv_list[0], recurrent_indices)),
        np.asarray(ssm_buf),
        np.asarray(conv_list[0]),
    )


@pytest.mark.parametrize("chunk", [False, True, "kernel"])
@pytest.mark.parametrize("dtype,tol", [(jnp.float32, 1e-5), (jnp.bfloat16, 2e-2)])
def test_kda_verify_matches_sequential_decode_and_commits_accepted_prefix(
    dtype, tol, chunk, monkeypatch
):
    if chunk == "kernel":
        # SGLJAX_KDA_VERIFY_KERNEL=1: multi-token Pallas recurrence (interpreted on CPU).
        import sgl_jax.srt.layers.attention.linear.kda_backend as kb

        monkeypatch.setattr(kb, "_VERIFY_KERNEL", True)
        monkeypatch.setenv("PALLAS_INTERPRET", "1")
        tol = max(tol, 2e-4)
    elif chunk:
        # SGLJAX_KDA_VERIFY_CHUNK=1 path: chunked prefill kernel (interpreted on CPU).
        import sgl_jax.srt.layers.attention.linear.kda_backend as kb

        monkeypatch.setattr(kb, "_VERIFY_CHUNK", True)
        monkeypatch.setenv("SGLANG_JAX_KDA_PREFILL_KERNEL", "chunked")
        monkeypatch.setenv("PALLAS_INTERPRET", "1")
        tol = max(tol, 2e-4)
    B = 3
    fb, pool, layer, ridx, toks = _setup(B, dtype)
    initial = _snapshot(pool, ridx)

    # Reference: T sequential single-token decode steps, snapshotting state after each.
    ref_out, ref_states = [], [initial]
    _set_meta(fb, ForwardMode.DECODE, B, ridx, 1)
    for t in range(T):
        out, (ssm_buf, conv_list) = _call(layer, fb, pool, {n: x[:, t] for n, x in toks.items()})
        pool.replace_buffer(([ssm_buf], [conv_list]))
        ref_out.append(np.asarray(out, np.float32))
        ref_states.append(_snapshot(pool, ridx))
    ref_out = np.stack(ref_out, axis=1)  # [B, T, H*V]

    # Restore the pre-verify pool and run one TARGET_VERIFY window of T tokens per request.
    ssm0 = jax.device_put(jnp.asarray(initial[2]), pool.recurrent_sharding)
    conv0 = jax.device_put(jnp.asarray(initial[3]).astype(pool.conv_dtype), pool.conv_sharding)
    pool.replace_buffer(([ssm0], [[conv0]]))
    _set_meta(fb, ForwardMode.TARGET_VERIFY, B, ridx, T)
    flat = {n: x.reshape((B * T,) + x.shape[2:]) for n, x in toks.items()}
    out, (record, conv_list) = _call(layer, fb, pool, flat)
    assert isinstance(record, KDAVerifyRecord)
    out = np.asarray(out, np.float32).reshape(B, T, -1)
    err = np.abs(out - ref_out).max()
    print(f"\n[{jnp.dtype(dtype).name}] verify-vs-decode output max_abs={err:.3e}")
    np.testing.assert_allclose(out, ref_out, rtol=tol, atol=tol)
    # Verify itself must not touch the pool.
    np.testing.assert_array_equal(np.asarray(record.ssm_full), initial[2])
    np.testing.assert_array_equal(np.asarray(conv_list[0]), np.asarray(conv0))

    for steps in ([0, T, 2], [1, 3, T], [T, T, T]):
        n = _put(jnp.asarray(steps, jnp.int32), P("data"))
        ssm_full, (conv_full,) = jax.jit(commit_kda_verify)(record, n)
        got_ssm = np.asarray(gather_ssm(pool, ssm_full, ridx))
        got_conv = np.asarray(gather_conv(pool, conv_full, ridx))
        for r, s in enumerate(steps):
            want_ssm, want_conv = ref_states[s][0][r], ref_states[s][1][r]
            np.testing.assert_allclose(got_ssm[r], want_ssm, rtol=tol, atol=tol)
            np.testing.assert_allclose(
                got_conv[r].astype(np.float32), want_conv.astype(np.float32), rtol=tol, atol=tol
            )
        print(
            f"[{jnp.dtype(dtype).name}] commit steps={steps} ssm max_abs="
            f"{max(np.abs(got_ssm[r] - ref_states[s][0][r]).max() for r, s in enumerate(steps)):.3e}"
        )
        # Slot 0 is the dummy/padding slot and must stay untouched.
        np.testing.assert_array_equal(np.asarray(ssm_full)[0], initial[2][0])


def test_commit_recurrent_verify_rewrites_only_records():
    B = 2
    fb, pool, layer, ridx, toks = _setup(B, jnp.float32, seed=1)
    _set_meta(fb, ForwardMode.TARGET_VERIFY, B, ridx, T)
    flat = {n: x.reshape((B * T,) + x.shape[2:]) for n, x in toks.items()}
    _, (record, conv_list) = _call(layer, fb, pool, flat)
    other = jnp.zeros((3,))
    updates = {"token_to_kv_pool": other, "recurrent_state_pool": ([record], [conv_list])}
    n = _put(jnp.asarray([2, 4], jnp.int32), P("data"))
    out = commit_recurrent_verify(updates, n)
    (ssm,), ((conv,),) = out["recurrent_state_pool"]
    want_ssm, (want_conv,) = commit_kda_verify(record, n)
    np.testing.assert_array_equal(np.asarray(ssm), np.asarray(want_ssm))
    np.testing.assert_array_equal(np.asarray(conv), np.asarray(want_conv))
    assert out["token_to_kv_pool"] is other
    plain = {"recurrent_state_pool": ([jnp.ones(2)], [[jnp.ones(2)]])}
    assert commit_recurrent_verify(plain, n) is plain
    assert commit_recurrent_verify({"token_to_kv_pool": other}, n) == {"token_to_kv_pool": other}


def test_linear_metadata_supports_target_verify():
    backend = LinearRecurrentAttnBackend(mesh=mesh)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        dp_size=1,
        per_dp_bs_size=3,
        input_ids=np.zeros(3 * T, np.int32),
        recurrent_indices=np.array([1, 2, 0], np.int32),
        has_initial_state=np.ones(3, np.bool_),
        recurrent_track_indices=None,
        recurrent_track_mask=None,
    )
    meta = backend.get_forward_metadata(batch)
    np.testing.assert_array_equal(np.asarray(meta.cu_q_lens), np.arange(4) * T)
