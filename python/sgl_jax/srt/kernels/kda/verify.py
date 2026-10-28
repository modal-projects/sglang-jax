"""Pallas TPU multi-token KDA recurrence for speculative verify / commit (state resident in VMEM).

The DFlash verify runs T (= block size, 4-8) tokens per request through the gated delta rule. The
naive JAX recurrence (``naive_recurrent_kda``) round-trips the full [B, H, K, V] fp32 state through
HBM several times per token (v8a 226 profile, c32 k8: ~35 of 88 ms per verify step); the chunked
prefill kernel is worse at this shape (grid N x H x total-chunks, a chunk per 4-8-token request).
This kernel loads each request's state block once, applies the T steps in VMEM and writes the
per-token outputs (and optionally the final state) once.

Math per step (identical to ``naive_recurrent_kda``, fp32)::

    S   = S * exp(g_t)[:, None]
    res = v_t - k_t @ S
    S   = S + (beta_t * k_t)[:, None] * res[None, :]
    o_t = (q_t * scale) @ S

A step with g = 0, beta = 0 and k = 0 is an exact no-op, which the commit uses to stop at the
accepted position with static shapes.
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


def _kernel(q_ref, k_ref, v_ref, g_ref, b_ref, h0_ref, o_ref, *rest, T, scale, final):
    hT_ref = rest[0] if final else None
    S = h0_ref[0].astype(jnp.float32)  # [Hb, K, V]
    for t in range(T):  # T is small and static
        q = q_ref[0, t].astype(jnp.float32)  # [Hb, K]
        k = k_ref[0, t].astype(jnp.float32)
        v = v_ref[0, t].astype(jnp.float32)  # [Hb, V]
        g = g_ref[0, t].astype(jnp.float32)  # [Hb, K]
        beta = b_ref[0, t].astype(jnp.float32)  # [Hb]
        S = S * jnp.exp(g)[:, :, None]
        res = v - jnp.sum(k[:, :, None] * S, axis=1)  # [Hb, V]
        S = S + (beta[:, None] * k)[:, :, None] * res[:, None, :]
        o_ref[0, t] = jnp.sum((q * scale)[:, :, None] * S, axis=1).astype(o_ref.dtype)
    if final:
        hT_ref[0] = S.astype(hT_ref.dtype)


def kda_multistep(q, k, v, g, beta, h0, *, scale=None, output_final_state=False, heads_per_block=None,
                  interpret=None):
    """q, k, g: [B, T, H, K]; v: [B, T, H, V]; beta: [B, T, H]; h0: [B, H, K, V].

    Returns o [B, T, H, V] (v.dtype) and, if requested, the final state [B, H, K, V] (fp32).
    q and k are used as given (callers l2-normalize them); g is the activated log-space gate.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K**-0.5 if scale is None else float(scale)
    hb = heads_per_block or H
    assert H % hb == 0, (H, hb)
    if interpret is None:
        interpret = _interpret_default()
    tok = lambda last: pl.BlockSpec((1, T, hb, last), lambda b, h: (b, 0, h, 0))
    in_specs = [
        tok(K),  # q
        tok(K),  # k
        tok(V),  # v
        tok(K),  # g
        pl.BlockSpec((1, T, hb), lambda b, h: (b, 0, h)),  # beta
        pl.BlockSpec((1, hb, K, V), lambda b, h: (b, h, 0, 0)),  # h0
    ]
    out_specs = [tok(V)]
    out_shape = [jax.ShapeDtypeStruct((B, T, H, V), v.dtype)]
    if output_final_state:
        out_specs.append(pl.BlockSpec((1, hb, K, V), lambda b, h: (b, h, 0, 0)))
        out_shape.append(jax.ShapeDtypeStruct((B, H, K, V), jnp.float32))
    outs = pl.pallas_call(
        functools.partial(_kernel, T=T, scale=scale, final=output_final_state),
        grid=(B, H // hb),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        interpret=interpret,
    )(q, k, v, g, beta, h0)
    return (outs[0], outs[1]) if output_final_state else (outs[0], None)
