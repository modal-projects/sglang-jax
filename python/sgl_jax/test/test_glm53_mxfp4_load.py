"""GLM-5.3-Flash Quark-MXFP4: build the real model class from a 2-layer config and load real
checkpoint tensors through the real weight loader (CPU, 8 virtual devices, TP8 / EP8 epmoe).

MXFP4_MINI points at a mini checkpoint cut from amd/GLM-5.3-Flash-Quark-MXFP4 (real tensors and
names; checkpoint layer 2 = KDA + dense FP8 MLP -> layer 0, layer 3 = DSA + MXFP4 MoE -> layer 1,
the first E routed experts, router rows sliced to E, a 1024-row vocab, and the checkpoint's
quantization_config with the same per-module scheme). The test fails on any checkpoint tensor the
mappings leave unread, on any model parameter left unloaded, and on any loaded value that differs
from an independent decode of the checkpoint bytes. A routed-expert forward through EPMoE (packed
FP4 gmm_v2 in interpret mode) is compared with a dense f32 reference.

    MXFP4_MINI=/path/to/mini python -m pytest -s sgl_jax/test/test_glm53_mxfp4_load.py
"""

import os
import subprocess
import sys

import pytest

MINI = os.environ.get("MXFP4_MINI")


def _run(mode: str, devices: int, marker: str):
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={devices}"
    r = subprocess.run([sys.executable, __file__, MINI, mode], env=env, capture_output=True, text=True)
    print(r.stdout[-20000:])
    print(r.stderr[-8000:])
    assert r.returncode == 0
    assert marker in r.stdout


@pytest.mark.skipif(not MINI, reason="set MXFP4_MINI to a mini Quark MXFP4 checkpoint")
def test_glm53_mxfp4_mini_load():
    _run("load", 8, "MXFP4_LOAD_OK")


@pytest.mark.skipif(not MINI, reason="set MXFP4_MINI to a mini Quark MXFP4 checkpoint")
def test_glm53_mxfp4_mini_epmoe_forward():
    _run("forward", 1, "MXFP4_FORWARD_OK")


@pytest.mark.skipif(not MINI, reason="set MXFP4_MINI to a mini Quark MXFP4 checkpoint")
def test_glm53_mxfp4_mini_epmoe_forward_fused_gate_up():
    _run("forward_fused", 1, "MXFP4_FORWARD_OK")


def _main(mini: str) -> None:
    import importlib
    import re

    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np
    from flax import nnx
    from safetensors import safe_open

    from sgl_jax.srt.configs.load_config import LoadConfig
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear
    from sgl_jax.srt.layers.moe import EPMoE
    from sgl_jax.srt.model_loader.loader import JAXModelLoader
    from sgl_jax.srt.model_loader.weights import loader as wl
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh
    from sgl_jax.srt.utils.quantization import mxfp4

    # gmm_v2's tiler on CPU: report a v6e chip.
    from jax.experimental.pallas import tpu as pltpu

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
    for name in ("jax._src.tpu_info", "jax._src.pallas.mosaic.tpu_info"):
        try:
            importlib.import_module(name).get_tpu_info = lambda: info
        except ImportError:
            pass
    pltpu.get_tpu_info = lambda: info
    tts = importlib.import_module("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.tuned_tile_sizes")
    tts.get_device_name = lambda num_devices=None: "TPU v6e"

    assert jax.device_count() == 8, jax.devices()
    mesh = create_device_mesh([1, 8], [1, 1], devices=jax.devices(), is_full_topology=True)

    # Capture every WeightLoader.load report and make checkpoint coverage strict.
    reports = []
    orig_load = wl.WeightLoader.load

    def strict_load(self, mappings, safetensors_partition=1, dummy=False, **_):
        rep = orig_load(self, mappings, safetensors_partition, dummy, validate_checkpoint_coverage=True)
        reports.append(rep)
        return rep

    wl.WeightLoader.load = strict_load

    mc = ModelConfig(model_path=mini, trust_remote_code=True, dtype="bfloat16", moe_backend="epmoe")
    qc = mc.quantization_config
    assert qc is not None and qc.quark_scheme is not None, "quark scheme not auto-detected"
    assert qc.moe_weight_format == "mxfp4" and qc.is_static_checkpoint
    print("quant config:", qc.to_dict()["quark_scheme"], "fp8 linears:", qc.linear_rules[0]["module_paths"])
    mc.ep_size = 8
    mc.hf_config.ep_size = 8
    mc.hf_config.moe_backend = mc.moe_backend.value
    mc.hf_config.use_dsa_sparse = True
    mc.configure_for_tensor_parallel(8)

    model = JAXModelLoader(LoadConfig(), mesh).load_model(mc)

    # ---- coverage: no unread checkpoint tensor, no skipped mapping input
    assert reports, "WeightLoader.load was not called"
    for rep in reports:
        assert not rep["unexpected"], f"unmapped checkpoint tensors: {rep['unexpected']}"
        assert not rep["skipped"], f"skipped mappings: {rep['skipped'][:20]}"
    st = os.path.join(mini, "model-00001-of-00001.safetensors")
    f = safe_open(st, framework="np")
    print("checkpoint tensors:", len(list(f.keys())), "loaded targets:", len(reports[0]["loaded"]))

    # ---- module types follow the checkpoint scheme
    layers = model.model.layers
    l0, l1 = layers[0], layers[1]
    fp8 = {
        "layer0.mlp.gate_proj": l0.mlp.gate_proj,
        "layer0.mlp.up_proj": l0.mlp.up_proj,
        "layer0.mlp.down_proj": l0.mlp.down_proj,
        "layer1.q_a_proj": l1.self_attn.q_a_proj,
        "layer1.q_b_proj": l1.self_attn.q_b_proj,
        "layer1.kv_a_proj_with_mqa": l1.self_attn.kv_a_proj_with_mqa,
        "layer1.o_proj": l1.self_attn.o_proj,
    }
    for k, m in fp8.items():
        assert isinstance(m, QuantizedLinear), (k, type(m))
        assert m.weight_q.get_value().dtype == jnp.float8_e4m3fn, k
    for k, m in {
        "layer0.kda.q_proj": l0.self_attn.q_proj,
        "layer0.kda.o_proj": l0.self_attn.o_proj,
        "layer1.shared.gate_proj": l1.shared_experts.gate_proj,
        "layer1.shared.down_proj": l1.shared_experts.down_proj,
        "layer1.indexer.wq_b": l1.self_attn.indexer.wq_b,
    }.items():
        assert type(m) is LinearBase and m.weight.get_value().dtype == jnp.bfloat16, (k, type(m))
    assert isinstance(l1.mlp, EPMoE) and l1.mlp.fp4_packed
    assert l1.mlp.wi_0.get_value().dtype == jnp.uint32
    n_q = sum(isinstance(m, QuantizedLinear) for _, m in nnx.iter_graph(model))
    print("QuantizedLinear modules:", n_q)
    assert n_q == len(fp8), n_q

    H = "model.language_model.layers"
    g = lambda k: f.get_tensor(k)  # noqa: E731
    host = lambda a: np.asarray(jax.device_get(a))  # noqa: E731

    # ---- values: routed experts (packed FP4), bit-identical to an independent decode
    E = model.config.n_routed_experts
    for name, proj in (("wi_0", "gate_proj"), ("wi_1", "up_proj"), ("wo", "down_proj")):
        w = host(getattr(l1.mlp, name).get_value())
        s = host(getattr(l1.mlp, name + "_scale").get_value()).astype(np.float32)
        assert w.shape[0] == E and s.shape[0] == E
        for e in range(E):
            packed, codes = g(f"{H}.1.mlp.experts.{e}.{proj}.weight"), g(f"{H}.1.mlp.experts.{e}.{proj}.weight_scale")
            np.testing.assert_array_equal(
                mxfp4.packed_fp4_to_dense(w[e], s[e]), mxfp4.dequantize_mxfp4(packed, codes).T,
                err_msg=f"expert {e} {proj}",
            )
    # shared experts: exact bf16 dequant
    for proj in ("gate_proj", "up_proj", "down_proj"):
        got = host(getattr(l1.shared_experts, proj).weight.get_value()).astype(np.float32)
        want = mxfp4.dequantize_mxfp4(
            g(f"{H}.1.mlp.shared_experts.{proj}.weight"), g(f"{H}.1.mlp.shared_experts.{proj}.weight_scale")
        ).T
        np.testing.assert_array_equal(got, want, err_msg=f"shared {proj}")
    # FP8 linears: weight bytes unchanged, block scale expanded per output column
    for k, m in fp8.items():
        layer, mod = k.split(".", 1)
        src = {"layer0": f"{H}.0.", "layer1": f"{H}.1.self_attn."}[layer] + mod.replace("mlp.", "mlp.")
        wq = g(src + ".weight")
        got = host(m.weight_q.get_value())
        np.testing.assert_array_equal(got.view(np.uint8), wq.view(np.uint8), err_msg=k)
        sc = g(src + ".weight_scale")
        deq_ckpt = mxfp4.dequantize_fp8_block(wq, sc)
        ws = host(m.weight_scale.get_value()).astype(np.float32)  # [in_blocks, 1, out]
        deq_model = got.astype(np.float32).reshape(got.shape[0], -1, 128) * np.transpose(ws, (2, 0, 1))[: got.shape[0]]
        np.testing.assert_array_equal(deq_model.reshape(got.shape), deq_ckpt, err_msg=k)
    # bf16 tensors (KDA projection, router): unchanged
    np.testing.assert_array_equal(
        host(l0.self_attn.q_proj.weight.get_value()).view(np.uint16), g(f"{H}.0.self_attn.q_proj.weight").view(np.uint16).T
    )
    print("value checks: routed experts (packed FP4), shared (bf16), FP8 linears, bf16 tensors OK")

    print("MXFP4_LOAD_OK")


def _forward(mini: str, fused: bool = False) -> None:
    """Routed experts of the mini checkpoint through EPMoE on one CPU device: weights placed by the
    loader's MXFP4 recipe, packed-FP4 gmm_v2 in interpret mode, vs a dense f64 reference. (One
    device: the Pallas interpreter's host callbacks deadlock under multi-device CPU execution; EP8
    sharding of the same module is covered by the v6e-8 AOT compile.)"""
    import functools
    import importlib

    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np
    from flax import nnx
    from jax.experimental.pallas import tpu as pltpu
    from safetensors import safe_open

    from sgl_jax.srt.layers.moe import EPMoE
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh
    from sgl_jax.srt.utils.quantization import mxfp4
    from sgl_jax.srt.utils.quantization.quark import quark_quantization_config

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
    for name in ("jax._src.tpu_info", "jax._src.pallas.mosaic.tpu_info"):
        try:
            importlib.import_module(name).get_tpu_info = lambda: info
        except ImportError:
            pass
    pltpu.get_tpu_info = lambda: info
    tts = importlib.import_module("sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.tuned_tile_sizes")
    tts.get_device_name = lambda num_devices=None: "TPU v6e"

    import json

    cfg = json.load(open(os.path.join(mini, "config.json")))
    tc = cfg["text_config"]
    E, K, Hd, Id, lim = (tc["n_routed_experts"], tc["num_experts_per_tok"], tc["hidden_size"],
                         tc["moe_intermediate_size"], tc["swiglu_limit"])
    qc = quark_quantization_config(cfg["quantization_config"])
    mesh = create_device_mesh([1, 1], [1, 1], devices=jax.devices()[:1], is_full_topology=True)
    with jax.set_mesh(mesh):
        moe = EPMoE(hidden_size=Hd, num_experts=E, num_experts_per_tok=K, intermediate_dim=Id,
                    mesh=mesh, ep_size=1, activation="silu_clamp", swiglu_limit=lim, layer_id=1,
                    quantization_config=qc)
        moe.quantize_weights(is_static=True)
    f = safe_open(os.path.join(mini, "model-00001-of-00001.safetensors"), framework="np")
    H = "model.language_model.layers.1.mlp.experts"
    deq = {}
    for name, proj in (("wi_0", "gate_proj"), ("wi_1", "up_proj"), ("wo", "down_proj")):
        inputs = []
        for e in range(E):
            inputs += [f.get_tensor(f"{H}.{e}.{proj}.weight"), f.get_tensor(f"{H}.{e}.{proj}.weight_scale")]
        w_spec = ("expert", "tensor", None) if name == "wo" else ("expert", None, "tensor")
        w, s = mxfp4.load_packed_fp4_experts(
            inputs, mesh=moe.moe_mesh, weight_spec=w_spec,
            scale_spec=tuple(moe._fp4_scale_spec(name + "_scale")), expert_ids=np.arange(E), key=proj)
        getattr(moe, name).set_value(w)
        getattr(moe, name + "_scale").set_value(s)
        deq[proj] = [mxfp4.dequantize_mxfp4(inputs[2 * e], inputs[2 * e + 1]).T.astype(np.float64)
                     for e in range(E)]
    if fused:
        os.environ["SGLJAX_MXFP4_FUSED_GATE_UP"] = "1"
        with jax.set_mesh(mesh):
            assert moe.fuse_gate_up(), "packed-FP4 gate/up fusion did not apply"
        assert moe.wi_1 is None and moe.wi_0.get_value().shape == (E, Hd // 8, 2 * Id)
    rng = np.random.default_rng(0)
    T = 8
    x = (rng.standard_normal((T, Hd)) * 0.5).astype(ml_dtypes.bfloat16)
    ids = np.stack([rng.choice(E, K, replace=False) for _ in range(T)]).astype(np.int32)
    wts = rng.random((T, K)).astype(np.float32)
    graphdef, state = nnx.split(moe)
    run = jax.jit(lambda st, a, b, c: nnx.merge(graphdef, st)(a, b, c))
    with jax.set_mesh(mesh):
        out = np.asarray(run(state, jnp.asarray(x), jnp.asarray(wts), jnp.asarray(ids))).astype(np.float64)
    xr = x.astype(np.float64)
    ref = np.zeros_like(out)
    for t in range(T):
        for j in range(K):
            e = ids[t, j]
            gt = np.minimum(xr[t] @ deq["gate_proj"][e], lim)
            up = np.clip(xr[t] @ deq["up_proj"][e], -lim, lim)
            h = (gt / (1 + np.exp(-gt))) * up
            ref[t] += wts[t, j] * (h @ deq["down_proj"][e])
    err = np.abs(out - ref)
    rel = err.max() / np.abs(ref).max()
    print(f"EPMOE_FORWARD fused_gate_up={fused} tokens={T} topk={K} experts={E} max_abs={err.max():.3e} "
          f"rel(of max)={rel:.3e} ref_max={np.abs(ref).max():.3e}")
    # bf16 GEMM1 outputs / bf16 GLU / bf16 output rounding only (weights exact)
    assert rel < 2e-2, rel
    print("MXFP4_FORWARD_OK")


if __name__ == "__main__":
    modes = {"load": _main, "forward": _forward, "forward_fused": lambda p: _forward(p, fused=True)}
    modes[sys.argv[2] if len(sys.argv) > 2 else "load"](sys.argv[1])
