"""Parity of the Pallas KDA decode kernel (``kda_decode_pool``) vs ``naive_recurrent_kda``.

The reference is the production naive decode path: gather ``pool[idx]`` ->
``has_initial_state`` mask -> ``naive_recurrent_kda`` (T=1) -> idx==0-guarded
scatter -> track-slot scatter. Checked: outputs, every pool slot (touched slots,
track slots, untouched slots, dummy slot 0), several B / H / head-block sizes,
bf16 and fp32 q/k/v, fp32 and bf16 state pools, both column-broadcast modes,
and the ``shard_map`` wrapper used by ``KDAAttnBackend``.

CPU runs use Pallas TPU interpret mode. Run with ``-s`` to see error stats.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.kda.decode import kda_decode_pool, kda_decode_pool_reference

K = V = 128

INTERPRET = jax.default_backend() != "tpu"


def _l2(x):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)


def _make_case(B, H, S, dtype, seed=0, pool_dtype=jnp.float32, pad_rows=1, track=False):
    rng = np.random.default_rng(seed)
    q = jnp.asarray(_l2(rng.standard_normal((B, H, K))), dtype)
    k = jnp.asarray(_l2(rng.standard_normal((B, H, K))), dtype)
    v = jnp.asarray(rng.standard_normal((B, H, V)), dtype)
    # activated log-gate: -exp(A) * softplus(.), a wide range of decays
    a_scale = np.exp(rng.uniform(-3, 1, (1, H, 1)))
    g = jnp.asarray(-a_scale * np.log1p(np.exp(rng.standard_normal((B, H, K)))), dtype)
    beta = jnp.asarray(1.0 / (1.0 + np.exp(-rng.standard_normal((B, H)))), dtype)
    pool = jnp.asarray(rng.standard_normal((S, H, K, V)) * 0.2, jnp.float32).astype(pool_dtype)
    pool = pool.at[0].set(0)  # dummy slot is never written -> zeros
    slots = rng.permutation(np.arange(1, S))
    idx = slots[:B].astype(np.int32)
    if pad_rows:
        idx[B - pad_rows :] = 0  # padding rows -> dummy slot
    has_init = rng.random(B) > 0.25
    has_init[0] = True
    kw = dict(has_initial_state=jnp.asarray(has_init))
    if track:
        tidx = np.zeros(B, np.int32)
        tmask = np.zeros(B, np.int32)
        spare = slots[B:]
        n_trk = min(len(spare), max(1, B // 2))
        rows = rng.choice(B - pad_rows, size=min(n_trk, B - pad_rows), replace=False)
        tidx[rows] = spare[: len(rows)]
        tmask[rows] = 1
        if len(rows) > 1:
            tmask[rows[-1]] = 0  # track idx set but mask off -> no write
        kw.update(track_idx=jnp.asarray(tidx), track_mask=jnp.asarray(tmask))
    return (q, k, v, g, beta, pool, jnp.asarray(idx)), kw


def _errs(a, b):
    a = np.asarray(jnp.asarray(a, jnp.float32))
    b = np.asarray(jnp.asarray(b, jnp.float32))
    abs_err = float(np.max(np.abs(a - b))) if a.size else 0.0
    rel_err = abs_err / max(float(np.max(np.abs(b))), 1e-30)
    return abs_err, rel_err


def _check(args, kw, *, tol_o, tol_s, label, **kernel_kw):
    q, k, v, g, beta, pool, idx = args
    scale = K**-0.5
    ref_o, ref_pool = jax.jit(
        lambda *a: kda_decode_pool_reference(*a, scale=scale, **kw)
    )(*args)
    o, new_pool = jax.jit(
        lambda *a: kda_decode_pool(*a, scale=scale, interpret=INTERPRET, **kw, **kernel_kw)
    )(*args)
    assert o.dtype == v.dtype and o.shape == v.shape
    assert new_pool.dtype == pool.dtype and new_pool.shape == pool.shape
    eo = _errs(o, ref_o)
    es = _errs(new_pool, ref_pool)
    print(
        f"[{label}] o: max_abs={eo[0]:.3e} max_rel={eo[1]:.3e} | "
        f"pool: max_abs={es[0]:.3e} max_rel={es[1]:.3e}"
    )
    assert eo[0] <= tol_o, (label, "o", eo)
    assert es[0] <= tol_s, (label, "pool", es)
    # untouched slots must be bit-identical (in-place aliasing writes only B rows)
    touched = set(np.asarray(idx).tolist()) | {0}
    if "track_idx" in kw:
        touched |= set(np.asarray(kw["track_idx"]).tolist())
    untouched = np.array(sorted(set(range(pool.shape[0])) - touched), np.int32)
    if untouched.size:
        np.testing.assert_array_equal(np.asarray(new_pool[untouched]), np.asarray(pool[untouched]))
    np.testing.assert_array_equal(np.asarray(new_pool[0]), np.asarray(pool[0]))
    return eo, es


# fp32 math in both paths; only reduction order differs -> ~1 ulp-level differences.
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize(
    "B,H,hb",
    [(1, 8, None), (3, 8, None), (5, 16, 8), (4, 32, 16), (2, 64, None)],
)
def test_decode_parity_fp32_state(B, H, hb, dtype):
    args, kw = _make_case(B, H, S=2 * B + 3, dtype=dtype, pad_rows=1 if B > 1 else 0)
    tol_o = 1e-5 if dtype == jnp.float32 else 1e-2  # bf16 output rounding
    label = f"B{B}_H{H}_hb{hb}_{jnp.dtype(dtype).name}"
    _check(args, kw, tol_o=tol_o, tol_s=1e-5, label=label, heads_per_block=hb)


@pytest.mark.parametrize("column_mode", ["mxu", "mxu_head", "xlu"])
def test_decode_column_modes(column_mode):
    args, kw = _make_case(3, 16, S=9, dtype=jnp.bfloat16, seed=1)
    _check(args, kw, tol_o=1e-2, tol_s=1e-5, label=f"cols_{column_mode}", column_mode=column_mode)


@pytest.mark.parametrize("B,H", [(4, 8), (6, 16)])
def test_decode_track_slots(B, H):
    args, kw = _make_case(B, H, S=3 * B + 2, dtype=jnp.float32, seed=2, track=True)
    _check(args, kw, tol_o=1e-5, tol_s=1e-5, label=f"track_B{B}_H{H}")


def test_decode_all_padding_and_no_initial_state():
    args, kw = _make_case(4, 8, S=6, dtype=jnp.float32, seed=3, pad_rows=2)
    kw["has_initial_state"] = jnp.zeros((4,), bool)
    _check(args, kw, tol_o=1e-5, tol_s=1e-5, label="no_init")
    args, kw = _make_case(3, 8, S=5, dtype=jnp.float32, seed=4, pad_rows=3)
    _check(args, kw, tol_o=1e-5, tol_s=1e-5, label="all_pad")


def test_decode_bf16_state_pool():
    # Both paths compute in fp32 and round the new state to bf16 once.
    args, kw = _make_case(5, 16, S=8, dtype=jnp.bfloat16, seed=5, pool_dtype=jnp.bfloat16)
    _check(args, kw, tol_o=1e-2, tol_s=2e-2, label="bf16_state")


def test_decode_sharded_pool_path():
    """shard_map wrapper used by KDAAttnBackend: per-rank local slot indices."""
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from sgl_jax.srt.layers.attention.linear.kda_backend import kda_decode_pool_sharded

    devs = jax.devices()
    tp = 2 if len(devs) >= 2 else 1
    dp = 2 if len(devs) >= 4 else 1
    mesh = Mesh(np.array(devs[: dp * tp]).reshape(dp, tp), ("data", "tensor"))
    B_local, H, S_local = 3, 16, 6
    per_rank = [
        _make_case(B_local, H, S=S_local, dtype=jnp.bfloat16, seed=10 + r, track=True)
        for r in range(dp)
    ]
    cat = lambda xs: jnp.concatenate(xs, axis=0)  # noqa: E731
    q, k, v, g, beta, pool, idx = (cat([c[0][i] for c in per_rank]) for i in range(7))
    hi = cat([c[1]["has_initial_state"] for c in per_rank])
    tidx = cat([c[1]["track_idx"] for c in per_rank])
    tmask = cat([c[1]["track_mask"] for c in per_rank])

    put = lambda x, spec: jax.device_put(x, NamedSharding(mesh, spec))  # noqa: E731
    args = (
        put(q, P("data", "tensor", None)),
        put(k, P("data", "tensor", None)),
        put(v, P("data", "tensor", None)),
        put(g, P("data", "tensor", None)),
        put(beta, P("data", "tensor")),
        put(pool, P("data", "tensor", None, None)),
        put(idx, P("data")),
    )
    meta = dict(
        has_initial_state=put(hi, P("data")),
        track_indices=put(tidx, P("data")),
        track_mask=put(tmask, P("data")),
    )
    o, new_pool = jax.jit(
        lambda *a, **m: kda_decode_pool_sharded(
            mesh, *a, scale=K**-0.5, interpret=INTERPRET, **m
        )
    )(*args, **meta)
    for r, ((rq, rk, rv, rg, rb, rpool, ridx), rkw) in enumerate(per_rank):
        ref_o, ref_pool = kda_decode_pool_reference(
            rq, rk, rv, rg, rb, rpool, ridx, scale=K**-0.5, **rkw
        )
        eo = _errs(o[r * B_local : (r + 1) * B_local], ref_o)
        es = _errs(new_pool[r * S_local : (r + 1) * S_local], ref_pool)
        print(
            f"[sharded dp{dp} tp{tp} rank{r}] o: max_abs={eo[0]:.3e} max_rel={eo[1]:.3e} | "
            f"pool: max_abs={es[0]:.3e} max_rel={es[1]:.3e}"
        )
        assert eo[0] <= 1e-2 and es[0] <= 1e-5, (r, eo, es)


@pytest.mark.skipif(
    os.environ.get("KDA_DECODE_LONG_DRIFT", "") != "1",
    reason="long sequential drift study; set KDA_DECODE_LONG_DRIFT=1",
)
@pytest.mark.parametrize("decay", [0.2, 0.002], ids=["fast_decay", "slow_decay"])
def test_bf16_state_drift_long(decay):
    """bf16 vs fp32 state pool over many sequential decode steps (Pallas kernel).

    g = -decay * softplus(N(0,1)): per-step decay exp(g) ~0.87 (fast) / ~0.9986 (slow,
    long memory, where bf16 rounding of the state accumulates the most).
    """
    steps = int(os.environ.get("KDA_DECODE_DRIFT_STEPS", "1024"))
    B, H = 2, 8
    rng = np.random.default_rng(7)
    base = jnp.asarray(rng.standard_normal((B + 1, H, K, V)) * 0.2, jnp.float32).at[0].set(0)
    idx = jnp.arange(1, B + 1, dtype=jnp.int32)

    def l2(x):
        return x * jax.lax.rsqrt(jnp.sum(x * x, -1, keepdims=True) + 1e-6)

    def inputs(t):
        key = jax.random.fold_in(jax.random.PRNGKey(0), t)
        kq, kk, kv, kg, kb = jax.random.split(key, 5)
        q = l2(jax.random.normal(kq, (B, H, K))).astype(jnp.bfloat16)
        k = l2(jax.random.normal(kk, (B, H, K))).astype(jnp.bfloat16)
        v = jax.random.normal(kv, (B, H, V)).astype(jnp.bfloat16)
        g = (-decay * jax.nn.softplus(jax.random.normal(kg, (B, H, K)))).astype(jnp.bfloat16)
        beta = jax.nn.sigmoid(jax.random.normal(kb, (B, H))).astype(jnp.bfloat16)
        return q, k, v, g, beta

    def run(pool):
        def body(pool, t):
            q, k, v, g, beta = inputs(t)
            o, pool = kda_decode_pool(q, k, v, g, beta, pool, idx, interpret=INTERPRET)
            return pool, o.astype(jnp.float32)

        return jax.lax.scan(body, pool, jnp.arange(steps))

    p32, o32 = jax.jit(run)(base)
    p16, o16 = jax.jit(run)(base.astype(jnp.bfloat16))
    so = _errs(o16, o32)
    ss = _errs(p16[1:], p32[1:])
    o_rms = float(jnp.sqrt(jnp.mean((o16 - o32) ** 2)) / jnp.sqrt(jnp.mean(o32**2)))
    s_rms = float(
        jnp.sqrt(jnp.mean((p16[1:].astype(jnp.float32) - p32[1:]) ** 2))
        / jnp.sqrt(jnp.mean(p32[1:] ** 2))
    )
    tail = slice(steps - 64, steps)
    o_rms_tail = float(
        jnp.sqrt(jnp.mean((o16[tail] - o32[tail]) ** 2)) / jnp.sqrt(jnp.mean(o32[tail] ** 2))
    )
    print(
        f"[bf16 drift {steps} steps decay={decay}] o: max_abs={so[0]:.3e} max_rel={so[1]:.3e} "
        f"rms_rel={o_rms:.3e} rms_rel_last64={o_rms_tail:.3e} | final state: "
        f"max_abs={ss[0]:.3e} max_rel={ss[1]:.3e} rms_rel={s_rms:.3e}"
    )
    assert o_rms < 5e-2, o_rms
