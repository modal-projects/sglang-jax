"""Pallas TPU single-token KDA (gated delta rule) decode, in place on the state pool.

Replaces the decode path ``gather pool[idx] -> naive_recurrent_kda(T=1) ->
scatter pool.at[idx]`` with one kernel that DMAs each request's state straight
from its pool slot into VMEM, applies the recurrence, and DMAs the new state
back to the same slot (and, optionally, to its track slot). The pool is
input/output aliased, so only the B touched slots move: one HBM read and one
HBM write of the [B, H, K, V] state instead of the ~10 full passes of the
gather / where / recurrence / where / scatter chain.

Math (``naive_recurrent_kda`` at T=1, fp32 accumulation)::

    S1  = S * exp(g)[:, None]               # per-K-row decay
    res = v - sum_k k[k] * S1[k, :]         # prediction error
    S2  = S1 + (beta * k)[:, None] * res[None, :]
    o   = sum_k (q * scale)[k] * S2[k, :]

q and k are taken as given (the caller l2-normalizes them, as for the naive
path); g is the activated log-space gate.

Pool slot conventions (identical to ``KDAAttnBackend.get_state`` /
``set_ssm_state`` / ``set_ssm_track_state``):
  * ``slot_idx == 0`` is the per-rank dummy slot used by padding rows: it is
    neither read (state treated as zero; the dummy slot is never written, so it
    holds zeros) nor written.
  * ``has_initial_state == False`` -> initial state is zero (slot not read).
  * track rows (``track_idx != 0 and track_mask != 0``) additionally get the
    new state written into ``pool[track_idx]``.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _interpret_default() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")


_N_COLS = 4  # per-K column operands: exp(g), k, beta*k, q*scale


def default_heads_per_block(num_heads: int, state_dtype=jnp.float32) -> int:
    """Heads per grid step: ~1 MiB state DMA per step (16 fp32 / 32 bf16 heads)."""
    target = 16 * 4 // jnp.dtype(state_dtype).itemsize
    if num_heads <= target:
        return num_heads
    for hb in range(target, 7, -1):
        if num_heads % hb == 0 and hb % 8 == 0:
            return hb
    return num_heads


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def vmem_bytes_per_step(
    heads_per_block: int, k: int, v: int, state_dtype, column_mode: str = "mxu"
) -> int:
    """VMEM footprint of one grid step (explicit scratch + double-buffered blocks).

    Excludes Mosaic-internal temporaries (small: per-head [K, V] values)."""
    hb = heads_per_block
    st = jnp.dtype(state_dtype).itemsize
    total = 2 * 2 * hb * k * v * st  # in_buf[2] + out_buf[2]
    total += 2 * hb * (_N_COLS * k + v) * 4  # packed per-head operands (x2 pipeline)
    total += 2 * hb * v * 4  # o block (x2 pipeline)
    if column_mode in ("mxu", "mxu_head"):
        p = _round_up(hb, 16)
        total += 2 * 3 * p * hb * v * 2  # one-hot selector (bf16, x2 pipeline)
    if column_mode == "mxu":
        total += _N_COLS * k * hb * v * 4  # broadcast columns scratch (f32)
    return total


def _split3_bf16(x):
    """x (f32) == hi + mid + lo exactly, each part bf16 (any partial sum is exact in f32)."""
    hi = x.astype(jnp.bfloat16)
    r1 = x - hi.astype(jnp.float32)
    mid = r1.astype(jnp.bfloat16)
    lo = (r1 - mid.astype(jnp.float32)).astype(jnp.bfloat16)
    return hi, mid, lo


def _kda_decode_kernel(
    # scalar prefetch (SMEM)
    src_ref,  # [B] int32: slot to read, or -1 (zero initial state)
    dst_ref,  # [B] int32: slot to write, or -1 (no write)
    tdst_ref,  # [B] int32: track slot to write, or -1
    # VMEM blocks
    x_ref,  # [1, Hb, 4K + V] f32: [exp(g) | k | beta*k | q*scale | v]
    sel_ref,  # [3P, Hb*V] bf16 one-hot head selector (mxu mode; else unused)
    pool_in_ref,  # [S, H, K, V] HBM (aliased with pool_ref; unused directly)
    # outputs
    o_ref,  # [1, Hb, V] f32
    pool_ref,  # [S, H, K, V] HBM
    # scratch
    in_buf,  # [2, Hb, K, V] state dtype
    out_buf,  # [2, Hb, K, V] state dtype
    cols_ref,  # [4K, Hb*V] f32 (mxu mode)
    sem_in,  # DMA[2]
    sem_out,  # DMA[2]
    sem_trk,  # DMA[2]
    *,
    nhb: int,
    hb_size: int,
    n_steps: int,
    head_k: int,
    head_v: int,
    column_mode: str,
    reqs: int = 1,
):
    """One grid step = ``reqs`` requests x ``hb_size`` heads (scratch [2, reqs*hb, K, V]; request
    r of the step uses rows r*hb .. r*hb+hb-1). reqs > 1 amortizes the per-step overhead when a
    chip holds few heads (TP8: 8 heads = 256 KiB of bf16 state per request)."""
    del pool_in_ref
    K, V = head_k, head_v
    R = reqs
    i = pl.program_id(0)
    slot = i % 2

    def coords(j):
        return (j // nhb) * R, (j % nhb) * hb_size

    def read_copy(j, s, r, src):
        _, h0 = coords(j)
        return pltpu.make_async_copy(
            pool_ref.at[src, pl.ds(h0, hb_size)],
            in_buf.at[s, pl.ds(r * hb_size, hb_size)],
            sem_in.at[s, r],
        )

    def write_copy(j, s, r, dst, sem):
        _, h0 = coords(j)
        return pltpu.make_async_copy(
            out_buf.at[s, pl.ds(r * hb_size, hb_size)],
            pool_ref.at[dst, pl.ds(h0, hb_size)],
            sem.at[s, r],
        )

    def start_read(j, s):
        b0, _ = coords(j)
        for r in range(R):
            src = src_ref[b0 + r]

            @pl.when(src >= 0)
            def _():
                read_copy(j, s, r, src).start()

    def wait_writes(j, s):
        b0, _ = coords(j)
        for r in range(R):
            dst = dst_ref[b0 + r]
            tdst = tdst_ref[b0 + r]

            @pl.when(dst >= 0)
            def _():
                write_copy(j, s, r, dst, sem_out).wait()

            @pl.when(tdst >= 0)
            def _():
                write_copy(j, s, r, tdst, sem_trk).wait()

    @pl.when(i == 0)
    def _():
        start_read(0, 0)

    if n_steps > 1:

        @pl.when(i + 1 < n_steps)
        def _():
            start_read(i + 1, 1 - slot)

    b0, _ = coords(i)
    for r in range(R):
        src = src_ref[b0 + r]

        @pl.when(src >= 0)
        def _():
            read_copy(i, slot, r, src).wait()

        @pl.when(src < 0)
        def _():
            in_buf[slot, pl.ds(r * hb_size, hb_size)] = jnp.zeros(
                (hb_size,) + in_buf.shape[2:], in_buf.dtype
            )

    # out_buf[slot] was last used by step i-2: its writebacks must land first.
    if n_steps > 2:

        @pl.when(i >= 2)
        def _():
            wait_writes(i - 2, slot)

    # ---- per-K columns (K on sublanes, replicated along the V lanes) --------
    # Kept in the same basic block as the recurrence so the scheduler can overlap
    # the MXU/pop-bound column build with the VALU-bound head loop.
    #   "mxu":      one bf16 matmul of all heads against a one-hot head selector;
    #               exact because each f32 operand is split into three bf16 parts
    #               (hi + mid + lo == x, every partial sum exact in f32).
    #   "mxu_head": same, one matmul per head.
    #   "xlu":      transpose + lane-broadcast (XLU-bound; ~1.7x more bundles).
    if column_mode == "xpose":
        # State transposed in VMEM ([V, K]: K on lanes), so every per-K operand is ONE row vreg
        # broadcast over sublanes (no replicated column build), and the two K-contractions are
        # lane reductions. Same fp32 operation order per element as naive_recurrent_kda.
        for r in range(R):
            x = x_ref[r]  # [Hb, 4K + V]
            for h in range(hb_size):
                row = x[h : h + 1]
                eg, kk, bk, qs = (row[:, t * K : (t + 1) * K] for t in range(_N_COLS))  # [1, K]
                vcol = row[:, _N_COLS * K :].T  # [V, 1]
                sT = in_buf[slot, r * hb_size + h].astype(jnp.float32).T  # [V, K]
                s1 = sT * eg
                u = jnp.sum(s1 * kk, axis=1, keepdims=True)  # [V, 1]
                s2 = s1 + (vcol - u) * bk
                out_buf[slot, r * hb_size + h] = s2.T.astype(out_buf.dtype)
                o_ref[r, pl.ds(h, 1), :] = jnp.sum(s2 * qs, axis=1, keepdims=True).T
    for r in range(R if column_mode != "xpose" else 0):
        x = x_ref[r]  # [Hb, 4K + V]
        if column_mode in ("mxu", "mxu_head"):
            p = sel_ref.shape[0] // 3
            parts = []
            for part in _split3_bf16(x[:, : _N_COLS * K]):
                if p > hb_size:
                    part = jnp.concatenate(
                        [part, jnp.zeros((p - hb_size, _N_COLS * K), part.dtype)], axis=0
                    )
                parts.append(part)
            lhs = jnp.concatenate(parts, axis=0)  # [3P, 4K]

        if column_mode == "mxu_head":
            head_cols = {}

            def column(t, h):
                if h not in head_cols:
                    head_cols[h] = jax.lax.dot_general(
                        lhs,
                        sel_ref[:, pl.ds(h * V, V)],
                        (((0,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )  # [4K, V]
                return head_cols[h][t * K : (t + 1) * K]

        elif column_mode == "mxu":
            cols_ref[...] = jax.lax.dot_general(
                lhs,
                sel_ref[...],
                (((0,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )  # [4K, Hb*V]: cols[t*K + k, h*V + v] = x[h, t*K + k]

            def column(t, h):
                return cols_ref[pl.ds(t * K, K), pl.ds(h * V, V)]

        else:
            xT = [x[:, t * K : (t + 1) * K].T for t in range(_N_COLS)]  # [K, Hb] each

            def column(t, h):
                return xT[t][:, h : h + 1]

        # ---- recurrence (fp32, naive_recurrent_kda operation order) -------------
        v_blk = x[:, _N_COLS * K :]  # [Hb, V]
        for h in range(hb_size):
            s0 = in_buf[slot, r * hb_size + h].astype(jnp.float32)  # [K, V]
            s1 = s0 * column(0, h)  # decay
            u = jnp.sum(column(1, h) * s1, axis=0, keepdims=True)  # [1, V]  k^T S'
            res = v_blk[h : h + 1, :] - u
            s2 = s1 + column(2, h) * res  # + (beta k) (x) res
            out_buf[slot, r * hb_size + h] = s2.astype(out_buf.dtype)
            o_ref[r, pl.ds(h, 1), :] = jnp.sum(column(3, h) * s2, axis=0, keepdims=True)

    for r in range(R):
        dst = dst_ref[b0 + r]
        tdst = tdst_ref[b0 + r]

        @pl.when(dst >= 0)
        def _():
            write_copy(i, slot, r, dst, sem_out).start()

        @pl.when(tdst >= 0)
        def _():
            write_copy(i, slot, r, tdst, sem_trk).start()

    @pl.when(i == n_steps - 1)
    def _():
        wait_writes(i, slot)
        if n_steps > 1:
            wait_writes(i - 1, 1 - slot)


def _head_selector(hb: int, v: int):
    """[3P, Hb*V] bf16: sel[part*P + h', h*V + c] = (h' == h)."""
    p = _round_up(hb, 16)
    rows = jnp.arange(3 * p) % p
    cols = jnp.arange(hb * v) // v
    return (rows[:, None] == cols[None, :]).astype(jnp.bfloat16)


def kda_decode_pool(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    pool: jax.Array,
    slot_idx: jax.Array,
    *,
    scale: float | None = None,
    has_initial_state: jax.Array | None = None,
    track_idx: jax.Array | None = None,
    track_mask: jax.Array | None = None,
    heads_per_block: int | None = None,
    column_mode: str = "mxu",
    interpret: bool | None = None,
) -> tuple[jax.Array, jax.Array]:
    """One KDA decode step for B requests, reading/writing ``pool`` in place.

    Args (all per-shard / local; call inside ``shard_map``):
      q, k:      [B, H, K] (already l2-normalized)
      v:         [B, H, V]
      g:         [B, H, K] activated log-gate
      beta:      [B, H]
      pool:      [S, H, K, V] recurrent state pool (fp32 or bf16), donated/aliased
      slot_idx:  [B] int32 pool slot per request (0 = padding / dummy slot)
      has_initial_state: [B] bool or None (all True)
      track_idx, track_mask: [B] int32 or None (no track writes)

    Returns (o [B, H, V] in v.dtype, new_pool).
    """
    B, H, K = q.shape
    V = v.shape[-1]
    assert k.shape == (B, H, K) and g.shape == (B, H, K), (q.shape, k.shape, g.shape)
    assert v.shape == (B, H, V) and beta.shape == (B, H), (v.shape, beta.shape)
    assert pool.ndim == 4 and pool.shape[1:] == (H, K, V), (pool.shape, (H, K, V))
    column_mode = os.environ.get("SGLJAX_KDA_DECODE_COLUMNS", column_mode)
    assert column_mode in ("mxu", "mxu_head", "xlu", "xpose"), column_mode
    assert K % 128 == 0 and V % 128 == 0, f"head dims must be lane-aligned, got K={K} V={V}"
    if scale is None:
        scale = K**-0.5
    if interpret is None:
        interpret = _interpret_default()
    hb = heads_per_block or default_heads_per_block(H, pool.dtype)
    assert H % hb == 0 and (hb % 8 == 0 or hb == H), (H, hb)
    nhb = H // hb
    # SGLJAX_KDA_DECODE_REQS: requests per grid step (largest divisor of B <= the knob; default 1)
    want = max(1, int(os.environ.get("SGLJAX_KDA_DECODE_REQS", "1")))
    R = max(r for r in range(1, min(want, B) + 1) if B % r == 0)
    n_steps = (B // R) * nhb

    # Same fp32 elementwise prologue as naive_recurrent_kda; one small fused pass.
    f32 = jnp.float32
    kf = k.astype(f32)
    x = jnp.concatenate(
        [
            jnp.exp(g.astype(f32)),
            kf,
            beta.astype(f32)[..., None] * kf,
            q.astype(f32) * scale,
            v.astype(f32),
        ],
        axis=-1,
    )  # [B, H, 4K + V]

    slot_idx = slot_idx.astype(jnp.int32)
    valid = slot_idx != 0
    fetch = valid if has_initial_state is None else valid & has_initial_state.astype(bool)
    src = jnp.where(fetch, slot_idx, -1).astype(jnp.int32)
    dst = jnp.where(valid, slot_idx, -1).astype(jnp.int32)
    if track_idx is None:
        tdst = jnp.full((B,), -1, jnp.int32)
    else:
        tvalid = track_idx != 0
        if track_mask is not None:
            tvalid = tvalid & (track_mask != 0)
        tdst = jnp.where(tvalid, track_idx, -1).astype(jnp.int32)

    if column_mode in ("mxu", "mxu_head"):
        sel = _head_selector(hb, V)
        cols_scratch = pltpu.VMEM(
            (_N_COLS * K, hb * V) if column_mode == "mxu" else (8, 128), f32
        )
    else:
        sel = jnp.zeros((16, 128), jnp.bfloat16)  # unused
        cols_scratch = pltpu.VMEM((8, 128), f32)  # unused

    def blk(i, *_):
        return (i // nhb, i % nhb, 0)

    kernel = functools.partial(
        _kda_decode_kernel,
        nhb=nhb,
        hb_size=hb,
        n_steps=n_steps,
        head_k=K,
        head_v=V,
        column_mode=column_mode,
        reqs=R,
    )
    vmem = vmem_bytes_per_step(hb, K, V, pool.dtype, column_mode) * R
    o, new_pool = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(n_steps,),
            in_specs=[
                pl.BlockSpec((R, hb, _N_COLS * K + V), blk),
                pl.BlockSpec(sel.shape, lambda i, *_: (0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=[
                pl.BlockSpec((R, hb, V), blk),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            scratch_shapes=[
                pltpu.VMEM((2, R * hb, K, V), pool.dtype),
                pltpu.VMEM((2, R * hb, K, V), pool.dtype),
                cols_scratch,
                pltpu.SemaphoreType.DMA((2, R)),
                pltpu.SemaphoreType.DMA((2, R)),
                pltpu.SemaphoreType.DMA((2, R)),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, H, V), f32),
            jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        ],
        # operand indices count the 3 scalar-prefetch args: pool is operand 5.
        input_output_aliases={5: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=max(32 << 20, vmem + (16 << 20)),
        ),
        interpret=interpret,
        name=f"kda_decode_h{H}_hb{hb}_{column_mode}" + (f"_r{R}" if R > 1 else ""),
    )(src, dst, tdst, x, sel, pool)
    return o.astype(v.dtype), new_pool


def kda_decode_pool_reference(
    q, k, v, g, beta, pool, slot_idx, *, scale=None, has_initial_state=None,
    track_idx=None, track_mask=None,
):
    """Pure-JAX equivalent of the naive pool path (gather -> naive -> scatter)."""
    from sgl_jax.srt.kernels.kda.naive import naive_recurrent_kda

    h0 = pool[slot_idx]
    if has_initial_state is not None:
        h0 = jnp.where(has_initial_state[:, None, None, None], h0, 0.0)
    o, s = naive_recurrent_kda(
        q[:, None], k[:, None], v[:, None], g[:, None], beta[:, None],
        scale=scale, initial_state=h0, output_final_state=True,
    )
    keep = (slot_idx == 0).reshape(-1, 1, 1, 1)
    new_pool = pool.at[slot_idx].set(jnp.where(keep, pool[slot_idx], s.astype(pool.dtype)))
    if track_idx is not None:
        tkeep = (track_idx == 0)
        if track_mask is not None:
            tkeep = tkeep | (track_mask == 0)
        tkeep = tkeep.reshape(-1, 1, 1, 1)
        new_pool = new_pool.at[track_idx].set(
            jnp.where(tkeep, new_pool[track_idx], s.astype(pool.dtype))
        )
    return o[:, 0], new_pool
