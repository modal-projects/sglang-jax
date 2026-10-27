"""AOT compile (libtpu compile-only, no TPU) of the GLM-5.3 MXFP4 routed-expert path for v6e.

Cases (all at GLM-5.3 shapes: H=4096, I=2048, 288 experts, top-8; EP8 -> 36 local experts):
  gmm/*    one chip (v6e:1x1): the EPMoE per-device GEMMs through the real megablox backend with
           rhs_fp4_packed -- gate and up [36, 4096/8 u32, 2048] (two calls, EPMoE's layout), the
           fused gate_up variant [36, 4096/8, 4096], and down [36, 2048/8, 4096]; lhs = M*8 sorted
           rows with group_offset (EP rank 7, all 288 group sizes).
  epmoe/*  v6e-8 (v6e:2x4): the full EPMoE module (__call__: shard_map over expert=8, permute,
           gate/up/clamp-SwiGLU/down packed-FP4 GMMs, unpermute, psum) built exactly as
           Glm5NextDecoderLayer builds it from the Quark quantization config (TP8 model mesh).
M (tokens) in --ms. Emits JSONL (mb_common.Recorder).

  JAX_PLATFORMS=cpu PYTHONPATH=<worktree>/python python mb_mxfp4_aot.py --aot --out x.jsonl
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mb_common as mc  # noqa: E402

H, I, E_TOTAL, TOPK, EP = 4096, 2048, 288, 8, 8
E_LOCAL = E_TOTAL // EP


def parse():
    p = mc.base_parser(__doc__)
    p.add_argument("--ms", default="1,8,32,128,1024")
    p.add_argument("--parts", default="gmm,epmoe")
    p.add_argument("--scale-dtype", default=os.environ.get("SGLJAX_MXFP4_SCALE_DTYPE", "bfloat16"))
    a = p.parse_args()
    if not a.aot:
        p.error("this script is compile-only: pass --aot")
    mc.setup_env(a)
    os.environ["SGLJAX_MXFP4_SCALE_DTYPE"] = a.scale_dtype
    return a


def main():
    a = parse()
    import dataclasses

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    mc.patch_tpu_info_for_cpu()
    from sgl_jax.srt.kernels.gmm import megablox_gmm_backend as be
    from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel import gmm_v2 as g2
    from sgl_jax.srt.utils.jax_utils import compilation_target

    rec = mc.Recorder(a, "mb_mxfp4_aot")
    SD = jax.ShapeDtypeStruct
    sdt = jnp.dtype(a.scale_dtype)
    ms = [int(x) for x in a.ms.split(",")]
    parts = a.parts.split(",")

    def tiles_of(m_rows, k, n):
        lhs = SD((m_rows, k), jnp.bfloat16)
        rhs = SD((E_LOCAL, k // 8, n), jnp.uint32)
        sc = SD((E_LOCAL, k // 32, n), sdt)
        cfg = g2.make_gmm_configs(
            lhs, rhs, sc, None, SD((E_TOTAL,), jnp.int32), SD((1,), jnp.int32),
            tile_info=g2.calculate_tiling, vmem_limit_bytes=int(128 * 2**20 * 0.9),
            out_dtype=jnp.bfloat16, acc_dtype=jnp.float32, maybe_quantize_lhs=False,
            zero_initialize=True, rhs_fp4_packed=True,
        )
        t = cfg.tiles
        return f"{t.tile_m}x{t.tile_k}x{t.tile_n}"

    # ------------------------------------------------------------------ per-chip GMMs
    if "gmm" in parts:
        dev1 = mc.aot_devices(mc.V6E_1)
        mesh1 = Mesh(np.array(dev1[:1]), ("x",))
        for m in ms:
            rows = m * TOPK
            rows_pad = -(-rows // 32) * 32
            lim = 10.0

            def layer(x, gs, off, wg, sg, wu, su, wd, sd):
                kw = dict(group_sizes=gs, group_offset=off, preferred_element_type=jnp.bfloat16,
                          acc_dtype=jnp.float32, maybe_quantize_lhs=False, rhs_fp4_packed=True,
                          interpret=False)
                g = be.gmm(lhs=x, rhs=wg, rhs_scale=sg, zero_initialize=False, **kw)
                u = be.gmm(lhs=x, rhs=wu, rhs_scale=su, zero_initialize=False, **kw)
                h = jax.nn.silu(jnp.minimum(g, lim)) * jnp.clip(u, -lim, lim)
                return be.gmm(lhs=h, rhs=wd, rhs_scale=sd, zero_initialize=True, **kw)

            def gate_up_fused(x, gs, off, w, s):
                return be.gmm(lhs=x, rhs=w, rhs_scale=s, group_sizes=gs, group_offset=off,
                              preferred_element_type=jnp.bfloat16, acc_dtype=jnp.float32,
                              maybe_quantize_lhs=False, rhs_fp4_packed=True, interpret=False,
                              zero_initialize=False)

            common = (SD((rows, H), jnp.bfloat16), SD((E_TOTAL,), jnp.int32), SD((1,), jnp.int32))
            wi = (SD((E_LOCAL, H // 8, I), jnp.uint32), SD((E_LOCAL, H // 32, I), sdt))
            wo = (SD((E_LOCAL, I // 8, H), jnp.uint32), SD((E_LOCAL, I // 32, H), sdt))
            cases = [
                (f"gmm/layer_two_call/M{m}", layer, common + wi + wi + wo,
                 {"tiles_gate": tiles_of(rows_pad, H, I), "tiles_down": tiles_of(rows_pad, I, H)}),
                (f"gmm/gate_up_fused/M{m}", gate_up_fused,
                 common + (SD((E_LOCAL, H // 8, 2 * I), jnp.uint32), SD((E_LOCAL, H // 32, 2 * I), sdt)),
                 {"tiles": tiles_of(rows_pad, H, 2 * I)}),
            ]
            for name, fn, specs, extra in cases:
                if not rec.want(name):
                    continue
                case = {"name": name, "M": m, "rows": rows, "e_local": E_LOCAL, "scale_dtype": sdt.name,
                        "weight_bytes_per_layer_chip": int(E_LOCAL * 3 * H * I // 2 + E_LOCAL * 3 * H * I // 32 * sdt.itemsize), **extra}
                try:
                    with compilation_target(mesh1):
                        info = mc.aot_compile(jax.jit(fn), specs, mesh1)
                    rec.emit({**case, **info})
                except Exception as e:  # noqa: BLE001
                    rec.error(case, e)

    # ------------------------------------------------------------------ EPMoE module, v6e-8
    if "epmoe" in parts:
        from flax import nnx

        from sgl_jax.srt.layers.moe import EPMoE
        from sgl_jax.srt.utils.mesh_utils import create_device_mesh
        from sgl_jax.srt.utils.quantization.quark import quark_quantization_config

        dev8 = mc.aot_devices(mc.V6E_8)
        mesh = create_device_mesh([1, 8], [1, 1], devices=dev8, is_full_topology=True)
        mx = dict(dtype="fp4", qscheme="per_group", group_size=32, scale_format="e8m0", is_dynamic=False)
        qc = quark_quantization_config({"quant_method": "quark", "global_quant_config": {"weight": mx}})

        with compilation_target(mesh), jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
            moe = nnx.eval_shape(lambda: EPMoE(
                hidden_size=H, num_experts=E_TOTAL, num_experts_per_tok=TOPK, intermediate_dim=I,
                mesh=mesh, ep_size=EP, weight_dtype=jnp.bfloat16, dtype=jnp.bfloat16,
                activation="silu_clamp", swiglu_limit=10.0, layer_id=3, quantization_config=qc))
            moe.quantize_weights(is_static=True, abstract=True)
        gdef, state = nnx.split(moe)
        state_specs = jax.tree.map(lambda v: v, state)
        w_info = {k: (str(getattr(moe, k).get_value().shape), str(getattr(moe, k).get_value().dtype))
                  for k in ("wi_0", "wi_0_scale", "wo", "wo_scale")}
        for m in ms:
            name = f"epmoe/ep8_tp8/M{m}"
            if not rec.want(name):
                continue
            case = {"name": name, "M": m, "weights": w_info}

            def fwd(st, x, w, ids):
                mod = nnx.merge(gdef, st)
                return mod(x, w, ids)

            try:
                rep = NamedSharding(mesh, P())
                specs = (
                    state_specs,
                    SD((m, H), jnp.bfloat16, sharding=rep),
                    SD((m, TOPK), jnp.float32, sharding=rep),
                    SD((m, TOPK), jnp.int32, sharding=rep),
                )
                import time

                t0 = time.perf_counter()
                with compilation_target(mesh), jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
                    compiled = jax.jit(fwd).trace(*specs).lower().compile()
                info = {"aot": "ok", "compile_s": round(time.perf_counter() - t0, 3)}
                ma = compiled.memory_analysis()
                if ma is not None:
                    info["aot_temp_bytes"] = int(getattr(ma, "temp_size_in_bytes", -1))
                    info["aot_arg_bytes"] = int(getattr(ma, "argument_size_in_bytes", -1))
                hlo = compiled.as_text()
                info["n_tpu_custom_calls"] = hlo.count("custom_call_target=\"tpu_custom_call\"")
                info["n_host_callbacks"] = hlo.count("xla_python_cpu_callback") + hlo.count(
                    "xla_ffi_python_cpu_callback")
                rec.emit({**case, **info})
            except Exception as e:  # noqa: BLE001
                rec.error(case, e)
    sys.exit(1 if rec.close() else 0)


if __name__ == "__main__":
    main()
