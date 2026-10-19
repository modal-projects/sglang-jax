"""Microbench: V4 Flash routed-expert GMM per device, FP4 (E2M1 codes, K32 bf16 scales) vs the
default FP8 (E4M3, per-row f32 scale) vs bf16 weights. Shapes follow EP=16: 16 local experts,
H=4096, I=2048; tokens per device = batch * top_k(6) / 16, spread evenly over local experts.

    python bench_deepseek_v4_fp4_gmm.py
"""
import statistics
import time

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
from sgl_jax.srt.layers.deepseek_v4_moe_loader import mxfp4_pair_to_resident_fp4

E, H, I = 16, 4096, 2048
rng = np.random.default_rng(0)


def weights(k, n, fmt):
    if fmt == "bf16":
        return jnp.asarray(rng.standard_normal((E, k, n)).astype(np.float32) * 0.02, jnp.bfloat16), None
    if fmt == "fp8":
        w = jnp.asarray(rng.standard_normal((E, k, n)).astype(np.float32), jnp.float8_e4m3fn)
        return w, jnp.full((E, 1, 1, n), 0.01, jnp.float32)
    ws, ss = [], []
    for _ in range(E):
        packed = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
        codes = rng.integers(120, 130, size=(n, k // 32), dtype=np.uint8)
        w, s = mxfp4_pair_to_resident_fp4(packed, codes, key="b")
        ws.append(w)
        ss.append(s)
    return jnp.asarray(np.stack(ws)), jnp.asarray(np.stack(ss))


def timeit(f, *a, iters=50, reps=7):
    for _ in range(3):
        jax.block_until_ready(f(*a))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            o = f(*a)
        jax.block_until_ready(o)
        ts.append((time.perf_counter() - t0) / iters)
    return statistics.median(ts), (max(ts) - min(ts)) / statistics.median(ts)


print(jax.devices()[0].device_kind)
W = {fmt: {"wi": weights(H, I, fmt), "wo": weights(I, H, fmt)} for fmt in ("bf16", "fp8", "fp4")}
for batch in (1, 8, 32, 64, 128, 256):
    m = max(E, batch * 6 // 16)
    m -= m % E
    gs = jnp.full((E,), m // E, jnp.int32)
    x = jnp.asarray(rng.standard_normal((m, H)).astype(np.float32), jnp.bfloat16)
    h = jnp.asarray(rng.standard_normal((m, I)).astype(np.float32), jnp.bfloat16)
    row = [f"batch {batch:4d} (m/dev={m:4d}):"]
    for fmt in ("bf16", "fp8", "fp4"):
        kw = dict(group_sizes=gs, preferred_element_type=jnp.bfloat16, maybe_quantize_lhs=False)
        if fmt == "fp4":
            kw["rhs_e2m1_codes"] = True
        (wi, wis), (wo, wos) = W[fmt]["wi"], W[fmt]["wo"]
        # one expert FFN: two up GEMMs (w1, w3) + one down GEMM
        f = jax.jit(
            lambda x, h, wi, wis, wo, wos, kw=kw: (
                gmm(lhs=x, rhs=wi, rhs_scale=wis, **kw),
                gmm(lhs=x, rhs=wi, rhs_scale=wis, **kw),
                gmm(lhs=h, rhs=wo, rhs_scale=wos, **kw),
            )
        )
        t, sp = timeit(f, x, h, wi, wis, wo, wos)
        wbytes = 3 * E * H * I * {"bf16": 2, "fp8": 1, "fp4": 0.5}[fmt]
        row.append(f"{fmt} {t*1e6:7.1f}us ({wbytes/t/1e9:5.0f} GB/s wt, sp {100*sp:.0f}%)")
    print("  ".join(row), flush=True)
