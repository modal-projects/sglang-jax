#!/usr/bin/env python3
"""GLM-5.3 routed-expert layer on one v6e chip: packed MXFP4 vs fp8 / bf16 / int4 (gmm_v2 paths).

One "layer" = what EPMoE runs per device: gate GEMM + up GEMM (or one fused gate_up GEMM) ->
silu_clamp in XLA -> down GEMM, through the real megablox backend `gmm` (sgl_jax), with the lhs
holding ALL ranks' sorted rows (M tokens x top-8) and group_offset selecting this chip's experts,
exactly as EPMoE calls it under expert parallelism.

Layouts (H=4096, I=2048, 288 experts, top-8):
  ep8   36 local experts per chip (v6e-8 EP8)
  ep16  18 local experts per chip (v6e-16 EP16)
Routing (top-8, distinct experts per token): random (seeded) | distinct (token t slot j -> expert
(8t+j) mod 288: the most unique experts). The timed chip is the EP rank with the most unique local
experts (ties: most rows).

Formats:
  mxfp4          packed MXFP4 W4A16, uint32 [E, K/8, N] + bf16 power-of-two [E, K/32, N] scales
                 (rhs_fp4_packed); two GEMMs for gate/up like EPMoE's default
  mxfp4_fused    same, gate_up fused into one [E, K/8, 2N] GEMM (SGLJAX_MXFP4_FUSED_GATE_UP)
  fp8_w8a16      fp8 e4m3 weights, per-channel f32 scale, bf16 activations
  fp8b128_w8a16  fp8 e4m3 weights, 128-K block f32 scale (the FP8 checkpoint's expert format)
  bf16           bf16 weights
  int4_w4a16     int4 weights, per-channel scale, bf16 activations
  int4_w4a8      int4 weights, per-channel scale, gmm_v2 activation quantization (int8 on v6e)
The packed-FP4 formats run each M with --fp4-paths (decode = K32 block-dot path, fullk = full-K
prefill path; forced via SGLJAX_FP4_PATH) so the decode/prefill cutoff can be chosen from data:
the final `fp4_cutoff` records give, per layout/routing/format, the per-M winner and the
modelled per-layer time of the auto rule for each candidate cutoff and basis
(SGLJAX_FP4_FULLK_BASIS local = dsv4-fp4 rule over local groups, global = rows per expert).

Metrics per case: us per layer, touched weight bytes (+scales) of the experts that got rows,
effective GB/s, kernel path/tiles, and (TPU, --check-ms) rel. error vs a dequantized f32 reference.

  python mb_mxfp4_gemv.py --out r.jsonl                 # TPU timing
  python mb_mxfp4_gemv.py --cpu-smoke --out s.jsonl     # CPU: tiny shapes, interpret numerics
  python mb_mxfp4_gemv.py --aot --out a.jsonl           # compile-only for v6e:1x1
"""

from __future__ import annotations

import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mb_common as mc  # noqa: E402

TOPK = 8
SWIGLU_LIMIT = 10.0
FORMATS = ("mxfp4", "mxfp4_fused", "fp8_w8a16", "fp8b128_w8a16", "bf16", "int4_w4a16", "int4_w4a8")
MS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096"
CUTOFFS = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def parse():
    p = mc.base_parser(__doc__.split("\n")[0])
    p.add_argument("--layouts", default="ep8,ep16")
    p.add_argument("--ms", default=MS)
    p.add_argument("--routings", default="random,distinct")
    p.add_argument("--formats", default=",".join(FORMATS))
    p.add_argument("--fp4-paths", default="decode,fullk")
    p.add_argument("--check-ms", default="1,8,128,1024", help="Ms checked vs reference (random)")
    p.add_argument("--resume", action="store_true",
                   help="skip cases already recorded without error in --out")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    mc.setup_env(a)
    return a


def shapes(layout, smoke):
    """(E_total, E_local, H, I)."""
    if smoke:
        return {"ep8": (32, 4, 256, 128), "ep16": (32, 2, 256, 128)}[layout]
    return {"ep8": (288, 36, 4096, 2048), "ep16": (288, 18, 4096, 2048)}[layout]


def make_routing(kind, m, e_total, seed):
    import numpy as np
    k = min(TOPK, e_total)
    if kind == "random":
        rng = np.random.default_rng(seed + 1000 * m)
        return np.stack([rng.choice(e_total, k, replace=False) for _ in range(m)]).astype(np.int32)
    if kind == "distinct":
        return ((np.arange(m)[:, None] * k + np.arange(k)[None, :]) % e_total).astype(np.int32)
    raise ValueError(kind)


def pick_rank(topk, e_total, e_local):
    import numpy as np
    flat = topk.reshape(-1)
    n_ranks = e_total // e_local
    uniq = [len(np.unique(flat[(flat >= r * e_local) & (flat < (r + 1) * e_local)]))
            for r in range(n_ranks)]
    rows = [int(((flat >= r * e_local) & (flat < (r + 1) * e_local)).sum()) for r in range(n_ranks)]
    best = max(range(n_ranks), key=lambda r: (uniq[r], rows[r]))
    return best, uniq[best], rows[best]


def is_fp4(fmt):
    return fmt.startswith("mxfp4")


def bytes_per_expert(fmt, H, I):
    """Weight + scale bytes of one expert (gate + up + down)."""
    k_n = 3 * H * I
    return {
        "bf16": 2 * k_n,
        "fp8_w8a16": k_n + (2 * I + H) * 4,
        "fp8b128_w8a16": k_n + (H // 128 * 2 * I + I // 128 * H) * 4,
        "int4_w4a16": k_n // 2 + (2 * I + H) * 4,
        "int4_w4a8": k_n // 2 + (2 * I + H) * 4,
        "mxfp4": k_n // 2 + (H // 32 * 2 * I + I // 32 * H) * 2,
        "mxfp4_fused": k_n // 2 + (H // 32 * 2 * I + I // 32 * H) * 2,
    }[fmt]


# ------------------------------------------------------------------------------------ weights

def make_weights(fmt, key, e, k, n, mode):
    """(w, scale) in gmm operand layout for one [E, K, N] projection (ShapeDtypeStruct in aot)."""
    import jax
    import jax.numpy as jnp
    SD = jax.ShapeDtypeStruct
    if is_fp4(fmt):
        if mode == "aot":
            return SD((e, k // 8, n), jnp.uint32), SD((e, k // 32, n), jnp.bfloat16)
        k1, k2 = jax.random.split(key)
        w = jax.random.bits(k1, (e, k // 8, n), jnp.uint32)
        codes = jax.random.randint(k2, (e, k // 32, n), 118, 125)  # Quark GLM-5.3: 119..122
        return w, jnp.exp2((codes - 127 + 14).astype(jnp.float32)).astype(jnp.bfloat16)
    wdt = {"bf16": jnp.bfloat16, "fp8_w8a16": jnp.float8_e4m3fn, "fp8b128_w8a16": jnp.float8_e4m3fn,
           "int4_w4a16": jnp.int4, "int4_w4a8": jnp.int4}[fmt]
    if fmt == "bf16":
        sshape = None
    elif fmt == "fp8b128_w8a16":
        sshape = (e, k // 128, 1, n)
    else:
        sshape = (e, 1, 1, n)
    if mode == "aot":
        return SD((e, k, n), wdt), (SD(sshape, jnp.float32) if sshape else None)
    k1, k2 = jax.random.split(key)
    if fmt == "bf16":
        w = (jax.random.normal(k1, (e, k, n), jnp.float32) * 0.05).astype(jnp.bfloat16)
    elif fmt.startswith("fp8"):
        w = (jax.random.normal(k1, (e, k, n), jnp.float32) * 2.0).astype(jnp.float8_e4m3fn)
    else:
        w = jax.random.randint(k1, (e, k, n), -7, 8, jnp.int32).astype(jnp.int8).astype(jnp.int4)
    s = None if sshape is None else jax.random.uniform(k2, sshape, jnp.float32, 0.002, 0.01)
    return w, s


def dequant_expert(fmt, w, s):
    """One expert's [K, N] f32 weight from gmm operands (w: [K(/8), N], s: per-expert scale)."""
    import jax.numpy as jnp
    if is_fp4(fmt):
        lut = jnp.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], jnp.float32)
        k8, n = w.shape
        nc = k8 // 8
        wr = w.reshape(nc, 8, n)
        lo = jnp.stack([(wr >> (8 * b)) & 0xF for b in range(4)], axis=2).reshape(nc, 32, n)
        if os.environ.get("SGLJAX_FP4_PACK", "v1") == "v2":  # low nibble q<<1 | s
            lo = ((lo & 1) << 3) | (lo >> 1)
        hi = jnp.stack([(wr >> (8 * b + 4)) & 0xF for b in range(4)], axis=2).reshape(nc, 32, n)
        codes = jnp.concatenate([lo, hi], axis=1).reshape(k8 * 8, n)
        sc = jnp.repeat(s.astype(jnp.float32), 32, axis=0) * 2.0**-14
        return lut[codes.astype(jnp.int32)] * sc
    wf = w.astype(jnp.float32)
    if s is None:
        return wf
    if s.shape[0] > 1:  # [K/128, 1, N] block scale
        return wf * jnp.repeat(s[:, 0, :], wf.shape[0] // s.shape[0], axis=0)
    return wf * s[0]


# ------------------------------------------------------------------------------------ main

def main():
    a = parse()
    import jax
    import jax.numpy as jnp
    import numpy as np

    if a.cpu_smoke or a.aot:
        mc.patch_tpu_info_for_cpu()
    from sgl_jax.srt.kernels.gmm import megablox_gmm_backend as be
    from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel import gmm_v2 as g2
    from sgl_jax.srt.utils.jax_utils import compilation_target

    rec = mc.Recorder(a, "mb_mxfp4_gemv")
    mode = rec.mode
    smoke = a.cpu_smoke
    SD = jax.ShapeDtypeStruct
    ms = [int(x) for x in a.ms.split(",")]
    if smoke:
        ms = [m for m in ms if m <= 4][:2] or [1, 4]
    check_ms = set() if a.no_check else {int(x) for x in a.check_ms.split(",")}
    if smoke:
        check_ms = set(ms)
    interpret = None if smoke else False  # smoke: backend interpret paths on CPU
    done = set()
    if a.resume and os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r and r.get("name"):
                done.add(r["name"])
    aot_mesh = None
    if mode == "aot":
        from jax.sharding import Mesh
        aot_mesh = Mesh(np.array(mc.aot_devices(mc.V6E_1)[:1]), ("x",))
    results = []  # for the cutoff summary

    def kernel_path(rows, k, n, e_local, e_total):
        """(path, tiles) gmm_v2 picks for a packed-FP4 [E_local, K/8, N] call under the env."""
        lhs = SD((-(-rows // 32) * 32, k), jnp.bfloat16)
        cfg = g2.make_gmm_configs(
            lhs, SD((e_local, k // 8, n), jnp.uint32), SD((e_local, k // 32, n), jnp.bfloat16), None,
            SD((e_total,), jnp.int32), SD((1,), jnp.int32), tile_info=g2.calculate_tiling,
            vmem_limit_bytes=int(128 * 2**20 * 0.9), out_dtype=jnp.bfloat16, acc_dtype=jnp.float32,
            maybe_quantize_lhs=False, zero_initialize=True, rhs_fp4_packed=True)
        t = cfg.tiles
        path = "fullk" if t.tile_m > g2.FP4_PACKED_MAX_TILE_M else "decode"
        return path, f"{t.tile_m}x{t.tile_k}x{t.tile_n}"

    for layout in a.layouts.split(","):
        e_total, e_local, H, I = shapes(layout, smoke)
        for fmt in a.formats.split(","):
            fused = fmt == "mxfp4_fused"
            key = jax.random.PRNGKey(a.seed)
            k1, k2, k3 = jax.random.split(key, 3)
            try:
                w1, s1 = make_weights(fmt, k1, e_local, H, 2 * I, mode)  # gate | up columns
                w2, s2 = make_weights(fmt, k2, e_local, I, H, mode)
                if not fused:
                    if mode == "aot":
                        wg, sg = make_weights(fmt, k1, e_local, H, I, mode)
                        wu, su = wg, sg
                    else:
                        wg, wu = w1[..., :I], w1[..., I:]
                        sg = None if s1 is None else s1[..., :I]
                        su = None if s1 is None else s1[..., I:]
                    consts = (wg, sg, wu, su, w2, s2)
                else:
                    consts = (w1, s1, w2, s2)
                if mode != "aot":
                    jax.block_until_ready(consts)
            except Exception as e:  # noqa: BLE001
                rec.error({"name": f"{layout}/{fmt}/alloc", "layout": layout, "fmt": fmt}, e)
                continue

            gkw = dict(preferred_element_type=jnp.bfloat16, acc_dtype=jnp.float32,
                       interpret=interpret, maybe_quantize_lhs=(fmt == "int4_w4a8"))
            if is_fp4(fmt):
                gkw["rhs_fp4_packed"] = True

            def layer(c, x, gs, off, fused=fused, gkw=gkw):
                # scalar group_offset, as EPMoE passes it (gmm v1 on CPU requires a scalar)
                kw = dict(group_sizes=gs, group_offset=off[0], **gkw)
                if fused:
                    wgu, sgu, wd, sd = c
                    h = be.gmm(lhs=x, rhs=wgu, rhs_scale=sgu, zero_initialize=False, **kw)
                    g, u = h[:, :I], h[:, I:]
                else:
                    wg_, sg_, wu_, su_, wd, sd = c
                    g = be.gmm(lhs=x, rhs=wg_, rhs_scale=sg_, zero_initialize=False, **kw)
                    u = be.gmm(lhs=x, rhs=wu_, rhs_scale=su_, zero_initialize=False, **kw)
                gf = jnp.minimum(g.astype(jnp.float32), SWIGLU_LIMIT)
                uf = jnp.clip(u.astype(jnp.float32), -SWIGLU_LIMIT, SWIGLU_LIMIT)
                act = (gf * jax.nn.sigmoid(gf) * uf).astype(jnp.bfloat16)
                return be.gmm(lhs=act, rhs=wd, rhs_scale=sd, zero_initialize=True, **kw)

            def step(c, s):
                x, gs, off = s
                y = layer(c, x, gs, off)
                return (mc.fold(x, y), gs, off)

            paths = a.fp4_paths.split(",") if is_fp4(fmt) else ["-"]
            for path in paths:
                if is_fp4(fmt):
                    # "scalew": decode tiles with the weight-scaled full-K matmul (SGLJAX_FP4_DECODE_SCALEW)
                    # "<path>v2": the same path with the v2 nibble packing (SGLJAX_FP4_PACK=v2)
                    base = path[:-2] if path.endswith("v2") else path
                    os.environ["SGLJAX_FP4_PACK"] = "v2" if base != path else "v1"
                    os.environ["SGLJAX_FP4_PATH"] = "decode" if base in ("scalew", "bits", "dec16", "dec8") else base
                    os.environ["SGLJAX_FP4_DECODE_SCALEW"] = {"scalew": "1", "bits": "2"}.get(base, "0")
                    # "dec16": the K32 decode path with 16-row tiles (SGLJAX_FP4_DECODE_TILE_M)
                    os.environ["SGLJAX_FP4_DECODE_TILE_M"] = {"dec16": "16", "dec8": "8"}.get(base, "32")
                    jax.clear_caches()  # the path is chosen at trace time
                for m in ms:
                    rows = m * TOPK
                    for routing in a.routings.split(","):
                        if mode == "aot" and routing != a.routings.split(",")[0]:
                            continue  # routing only changes runtime scalars
                        name = f"{layout}/{fmt}/{path}/M{m}/{routing}"
                        if not rec.want(name) or name in done:
                            continue
                        topk = make_routing(routing, m, e_total, a.seed)
                        rank, uniq, lrows = pick_rank(topk, e_total, e_local)
                        lo = rank * e_local
                        flat = topk.reshape(-1)
                        order = np.argsort(flat, kind="stable")
                        tok = np.repeat(np.arange(m), TOPK)[order]
                        gs = np.bincount(flat, minlength=e_total).astype(np.int32)
                        touched = uniq * bytes_per_expert(fmt, H, I)
                        case = {"name": name, "layout": layout, "fmt": fmt, "fp4_path": path,
                                "M": m, "routing": routing, "rows": rows, "local_rows": lrows,
                                "e_local": e_local, "experts_touched": uniq, "ep_rank": rank,
                                "touched_bytes": int(touched),
                                "rows_per_expert_global": rows / e_total,
                                "rows_per_local_expert": lrows / e_local}
                        if is_fp4(fmt):
                            try:
                                n_gu = 2 * I if fused else I
                                case["path_taken"], case["tiles_gate_up"] = kernel_path(
                                    rows, H, n_gu, e_local, e_total)
                                case["path_taken_down"], case["tiles_down"] = kernel_path(
                                    rows, I, H, e_local, e_total)
                                os.environ["SGLJAX_FP4_PATH"] = "auto"
                                case["auto_path_default"] = kernel_path(rows, H, I, e_local, e_total)[0]
                            except Exception as e:  # noqa: BLE001
                                case["path_error"] = str(e)[:300]
                            finally:
                                os.environ["SGLJAX_FP4_PATH"] = "decode" if base in ("scalew", "bits", "dec16", "dec8") else base
                        if mode == "aot":
                            state = (SD((rows, H), jnp.bfloat16), SD((e_total,), jnp.int32),
                                     SD((1,), jnp.int32))
                        else:
                            x = jax.random.normal(jax.random.fold_in(k3, m), (m, H), jnp.float32)
                            x = (x * 0.5).astype(jnp.bfloat16)
                            xs = x[jnp.asarray(tok)]
                            state = (xs, jnp.asarray(gs), jnp.asarray([lo], jnp.int32))
                        # numerics (tpu / smoke) on the local rows
                        if mode != "aot" and m in check_ms and (routing == "random" or smoke):
                            try:
                                y = jax.jit(lambda c, s: layer(c, *s))(consts, state)
                                mlo = int(gs[:lo].sum())
                                y = np.asarray(y[mlo:mlo + lrows], np.float32)
                                ref = expert_ref(fmt, consts, fused, np.asarray(xs, np.float32),
                                                 gs, lo, e_local, I, dequant_expert)
                                den = float(np.abs(ref).max()) or 1.0
                                case["check_rel_err"] = float(np.abs(y - ref).max()) / den
                                # vs the reference rounded to the timed path's bf16 output
                                refb = ref.astype(jnp.bfloat16).astype(np.float32)
                                case["check_rel_err_bf16out"] = float(np.abs(y - refb).max()) / den
                            except Exception as e:  # noqa: BLE001
                                case["check_error"] = str(e)[:500]

                        def derive(us, touched=touched):
                            return {"us_per_layer": round(us, 3),
                                    "weight_GBps": round(touched / us / 1e3, 2)}

                        ctx = compilation_target(aot_mesh) if mode == "aot" else _null()
                        with ctx:
                            mc.run_case(rec, case, consts, state, step, derive=derive,
                                        key=(layout, fmt, path, m), mesh=aot_mesh)
                        results.append(case)
            os.environ.pop("SGLJAX_FP4_PATH", None)
            os.environ.pop("SGLJAX_FP4_DECODE_SCALEW", None)
            os.environ.pop("SGLJAX_FP4_DECODE_TILE_M", None)
            os.environ.pop("SGLJAX_FP4_PACK", None)
            del consts, w1, s1, w2, s2
            if not fused:
                del wg, sg, wu, su
            gc.collect()
            jax.clear_caches()
    if mode == "tpu":
        summarize_cutoffs(rec, a.out)
    sys.exit(1 if rec.close() else 0)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def expert_ref(fmt, consts, fused, xs, gs, lo, e_local, I, dequant):
    """Dequantized f32 reference for this chip's rows (act rounded to bf16 like the timed path)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    if fused:
        wgu, sgu, wd, sd = consts
    else:
        wg, sg, wu, su, wd, sd = consts
    deq = jax.jit(lambda w, s: dequant(fmt, w, s))
    outs = []
    r0 = int(gs[:lo].sum())
    for e in range(e_local):
        c = int(gs[lo + e])
        if c == 0:
            continue
        x = jnp.asarray(xs[r0:r0 + c])
        r0 += c
        pick = lambda s: None if s is None else s[e]  # noqa: E731
        if fused:
            wgu_e = deq(wgu[e], pick(sgu))
            h = x @ wgu_e
            g, u = h[:, :I], h[:, I:]
        else:
            g = x @ deq(wg[e], pick(sg))
            u = x @ deq(wu[e], pick(su))
        # the timed path's GEMM outputs are bf16; int4_w4a8's activation quantization error is
        # part of the measured difference
        g = g.astype(jnp.bfloat16).astype(jnp.float32)
        u = u.astype(jnp.bfloat16).astype(jnp.float32)
        g = jnp.minimum(g, SWIGLU_LIMIT)
        u = jnp.clip(u, -SWIGLU_LIMIT, SWIGLU_LIMIT)
        act = (g * jax.nn.sigmoid(g) * u).astype(jnp.bfloat16).astype(jnp.float32)
        outs.append(np.asarray(act @ deq(wd[e], pick(sd))))
    return np.concatenate(outs, axis=0)


def summarize_cutoffs(rec, out_path):
    """Per layout/routing/fp4 format: decode vs fullk per M, and the modelled auto-rule time for
    each candidate cutoff and basis (sum over the measured Ms, plus the per-M choice)."""
    import collections
    rows = [json.loads(l) for l in open(out_path)]
    rows = [r for r in rows if r.get("script") == "mb_mxfp4_gemv" and "us_per_layer" in r
            and str(r.get("fmt", "")).startswith("mxfp4")]
    groups = collections.defaultdict(dict)
    for r in rows:
        groups[(r["layout"], r["routing"], r["fmt"])].setdefault(r["M"], {})[r["fp4_path"]] = r
    for (layout, routing, fmt), by_m in sorted(groups.items()):
        per_m = []
        for m in sorted(by_m):
            d, f = by_m[m].get("decode"), by_m[m].get("fullk")
            if d is None or f is None:
                continue
            per_m.append({"M": m, "rows_per_expert_global": d["rows_per_expert_global"],
                          "rows_total": d["rows"], "e_local": d["e_local"],
                          "decode_us": d["us_per_layer"], "fullk_us": f["us_per_layer"],
                          "winner": "fullk" if f["us_per_layer"] < d["us_per_layer"] else "decode"})
        if not per_m:
            continue
        model = {}
        for basis in ("local", "global"):
            for c in CUTOFFS:
                tot = 0.0
                for p in per_m:
                    groups_n = p["e_local"] if basis == "local" else 288
                    use_f = p["rows_total"] >= c * groups_n
                    tot += p["fullk_us"] if use_f else p["decode_us"]
                model[f"{basis}:{c}"] = round(tot, 2)
        best = min(model, key=model.get)
        oracle = round(sum(min(p["decode_us"], p["fullk_us"]) for p in per_m), 2)
        rec.emit({"name": f"fp4_cutoff/{layout}/{fmt}/{routing}", "kind": "fp4_cutoff",
                  "layout": layout, "fmt": fmt, "routing": routing, "per_m": per_m,
                  "sum_us_by_rule": model, "best_rule": best, "best_sum_us": model[best],
                  "oracle_sum_us": oracle,
                  "note": "sum over measured Ms of us/layer under SGLJAX_FP4_FULLK_BASIS:"
                          "SGLJAX_FP4_FULLK_MIN_ROWS; per_m shows where fullk wins"})


if __name__ == "__main__":
    main()
