"""Unit tests for the DFlash2 draft port (tiny configs, CPU)."""

import json
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.mesh import AxisType
from jax.sharding import Mesh


def _mesh():
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    return Mesh(devices, ("data", "tensor"), axis_types=(AxisType.Explicit, AxisType.Explicit))


def _tiny_dflash2_config(**overrides):
    cfg = dict(
        architectures=["DFlash2DraftModel"],
        model_type="qwen3",
        vocab_size=96,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        rope_parameters={"rope_type": "default", "rope_theta": 2000000.0},
        rms_norm_eps=1e-5,
        attention_bias=False,
        layer_types=["sliding_attention", "sliding_attention"],
        sliding_window=64,
        is_causal=True,
        dflash_config={
            "block_size": 4,
            "conv_group_size": 8,
            "conv_kernel_size": 2,
            "mask_token_id": 95,
            "output_multiplier": 1.0,
            "selector_rank": 8,
            "selector_top_k": 5,
            "target_layer_ids": [0, 1, 2],
        },
        tie_word_embeddings=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_dflash2_architectures_are_registered():
    from sgl_jax.srt.models.registry import ModelRegistry

    for arch in ("DFlash2DraftModel", "MiniMaxDFlash2DraftModel", "DFlashDraftModel"):
        model_cls, resolved = ModelRegistry.resolve_model_cls([arch])
        assert resolved == arch and model_cls.__name__ == arch


def test_layer_attention_params_follow_upstream():
    from sgl_jax.srt.layers.radix_attention import AttentionType
    from sgl_jax.srt.models.dflash import get_dflash_layer_attention_params as params

    cfg = SimpleNamespace(layer_types=["sliding_attention", "full_attention"], sliding_window=64)
    assert params(cfg, 0) == (64, AttentionType.DECODER)
    assert params(cfg, 1) == (0, AttentionType.ENCODER_ONLY)
    cfg.is_causal = True
    assert params(cfg, 1) == (0, AttentionType.DECODER)
    cfg.is_causal = False
    assert params(cfg, 0) == (64, AttentionType.ENCODER_ONLY)
    assert params(SimpleNamespace(), 3) == (0, AttentionType.ENCODER_ONLY)  # classic DFlash


def test_model_options_reject_unported_features():
    from sgl_jax.srt.speculative.dflash_util import get_dflash_model_options

    opts = get_dflash_model_options(_tiny_dflash2_config())
    assert opts.has_conv and opts.has_selector and opts.block_size == 4
    bad = _tiny_dflash2_config()
    bad.dflash_config = dict(bad.dflash_config, attention_sink_bias=True)
    with pytest.raises(NotImplementedError):
        get_dflash_model_options(bad)
    bad.dflash_config = dict(bad.dflash_config, attention_sink_bias=False, conv_group_size=0)
    with pytest.raises(ValueError):
        get_dflash_model_options(bad)


def test_grouped_conv_matches_numpy_and_isolates_blocks():
    from sgl_jax.srt.models.dflash import DFlashGroupedConv

    mesh = _mesh()
    H, B, T, G = 16, 4, 3, 4
    rng = np.random.default_rng(0)
    with jax.set_mesh(mesh):
        conv = DFlashGroupedConv(H, B, T, G, mesh=mesh, dtype=jnp.float32)
        base = rng.normal(size=(2, T, H)).astype(np.float32)
        conv.base_kernel.value = jnp.asarray(base)
        x = rng.normal(size=(3 * B, H)).astype(np.float32)
        delta = rng.normal(size=(3 * B, T, H // G)).astype(np.float32)
        out = np.asarray(conv._convolve(jnp.asarray(x), jnp.asarray(delta), side=1))

    coeff = base[1][None] + np.repeat(delta, G, axis=-1)
    ref = np.zeros_like(x)
    for n in range(x.shape[0]):
        for t in range(T):
            if n % B >= t:  # never reads across a block boundary
                ref[n] += coeff[n, t] * x[n - t]
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def _tiny_model(dtype=jnp.float32):
    from sgl_jax.srt.models.dflash import DFlash2DraftModel

    mesh = _mesh()
    cfg = _tiny_dflash2_config()
    with jax.set_mesh(mesh):
        model = DFlash2DraftModel(cfg, mesh=mesh, dtype=dtype)
        # Non-trivial selector tables (zeros at init would make the pair term vanish).
        rng = np.random.default_rng(1)
        sel = model.candidate_selector
        sel.predecessor_codebook.value = jnp.asarray(rng.normal(size=(96, 8)), dtype)
        sel.successor_codebook.value = jnp.asarray(rng.normal(size=(96, 8)), dtype)
    return model, mesh, cfg


def test_dflash2_weight_mappings_cover_every_parameter():
    from flax import nnx

    model, _, cfg = _tiny_model()
    mappings = model._create_weight_mappings()
    targets = {spec.target_path for spec in mappings.values()}
    params = {
        ".".join(map(str, path)) for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param))
    }
    assert targets == params
    assert len(mappings) == 15 * cfg.num_hidden_layers + 6


def test_propose_block_tokens_under_jit_matches_eager_and_upstream_greedy():
    from flax import nnx

    from sgl_jax.srt.models.dflash import CandidateSelector

    model, mesh, cfg = _tiny_model()
    bs, e, H, V = 3, 3, cfg.hidden_size, cfg.vocab_size
    rng = np.random.default_rng(2)
    hidden = jnp.asarray(rng.normal(size=(bs, e, H)), jnp.float32)
    # Padded head: rows >= V must never be proposed, even with the largest logits.
    lm_head = np.asarray(rng.normal(size=(V + 32, H)), np.float32)
    lm_head[V:] = 50.0
    lm_head = jnp.asarray(lm_head)
    anchors = jnp.asarray([3, 7, 11], jnp.int32)

    graphdef, state = nnx.split(model)

    @jax.jit
    def run(state, hidden, lm_head, anchors):
        m = nnx.merge(graphdef, state)
        return m.propose_block_tokens(hidden, lm_head, anchors, vocab_size=V, return_lattice=True)

    with jax.set_mesh(mesh):
        tokens, ids, unary, scores = run(state, hidden, lm_head, anchors)
        eager_tokens = model.propose_block_tokens(hidden, lm_head, anchors, vocab_size=V)
    tokens, ids, scores = np.asarray(tokens), np.asarray(ids), np.asarray(scores)
    np.testing.assert_array_equal(tokens, np.asarray(eager_tokens))
    assert ids.max() < V and (tokens < V).all()

    # Upstream greedy sample_path: argmax of slot 0's anchor row, then follow argmax maps.
    idx = scores[:, 0, 0].argmax(-1)
    ref = [ids[np.arange(bs), 0, idx]]
    for slot in range(1, e):
        idx = scores[np.arange(bs), slot, idx].argmax(-1)
        ref.append(ids[np.arange(bs), slot, idx])
    np.testing.assert_array_equal(tokens, np.stack(ref, 1))

    # Lattice: unary + <A[pred] * W h, B[cand]>, pred = anchor for slot 0.
    sel = model.candidate_selector
    A, Bt = np.asarray(sel.predecessor_codebook.value), np.asarray(sel.successor_codebook.value)
    W = np.asarray(sel.hidden_projection.weight.value)
    proj = np.asarray(hidden) @ W
    pred = np.concatenate(
        [np.repeat(np.asarray(anchors)[:, None, None], ids.shape[-1], -1), ids[:, :-1]], axis=1
    )
    want = np.asarray(unary)[:, :, None, :] + np.einsum(
        "blpr,blcr->blpc", A[pred] * proj[:, :, None], Bt[ids]
    )
    np.testing.assert_allclose(scores, want, rtol=1e-4, atol=1e-4)
    assert CandidateSelector.greedy_path is not None


def test_candidate_topk_tensor_parallel_matches_global_topk():
    """Run with 4 virtual CPU devices (tp=4, dp=1) in a subprocess."""
    code = textwrap.dedent("""
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax, jax.numpy as jnp, numpy as np
        from jax._src.mesh import AxisType
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from sgl_jax.srt.models.dflash import candidate_topk
        mesh = Mesh(np.array(jax.devices()[:4]).reshape(1, 4), ("data", "tensor"),
                    axis_types=(AxisType.Explicit, AxisType.Explicit))
        rng = np.random.default_rng(0)
        V, Vp, k = 1000, 1024, 7
        x = rng.normal(size=(6, Vp)).astype(np.float32)
        x[:, V:] = 100.0  # padding columns must be excluded
        with jax.set_mesh(mesh):
            xs = jax.device_put(jnp.asarray(x), NamedSharding(mesh, P("data", "tensor")))
            ids, vals = jax.jit(lambda a: candidate_topk(a, k, vocab_size=V, mesh=mesh))(xs)
        ref_ids = np.argsort(-x[:, :V], axis=-1)[:, :k]
        assert (np.asarray(ids) == ref_ids).all(), (np.asarray(ids), ref_ids)
        np.testing.assert_allclose(np.asarray(vals), np.take_along_axis(x, ref_ids, -1))
        print("OK")
        """)
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-3000:]


def test_worker_resolves_dflash2_class_from_checkpoint_architecture(tmp_path):
    from sgl_jax.srt.models.dflash import DFlash2DraftModel, DFlashDraftModel
    from sgl_jax.srt.speculative.dflash_worker import DFlashWorker

    cfg = dict(vars(_tiny_dflash2_config()))
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    args = SimpleNamespace(
        speculative_draft_model_path=str(tmp_path), speculative_draft_model_revision=None
    )
    assert DFlashWorker._resolve_draft_model_class(args) is DFlash2DraftModel
    v1 = tmp_path / "v1"  # get_config caches per path
    v1.mkdir()
    cfg["architectures"] = ["DFlashDraftModel"]
    (v1 / "config.json").write_text(json.dumps(cfg))
    args.speculative_draft_model_path = str(v1)
    assert DFlashWorker._resolve_draft_model_class(args) is DFlashDraftModel


def test_set_block_size_reaches_every_conv():
    model, _, _ = _tiny_model()
    model.set_block_size(6)
    assert model.block_size == 6
    for layer in model.model.layers:
        assert layer.attention_conv.block_size == 6 and layer.mlp_conv.block_size == 6


def test_glm5_next_dflash_taps_are_hc_contracted_layer_outputs():
    from sgl_jax.srt.models.glm5_next import (
        Glm5NextForConditionalGeneration,
        hc_contract,
    )

    x = jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 12)  # hc=4 streams of D=3
    np.testing.assert_allclose(
        np.asarray(hc_contract(x, 4)), np.asarray(x).reshape(2, 4, 3).mean(1)
    )
    fake = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=45),
        model=SimpleNamespace(layers_to_capture=[]),
        capture_aux_hidden_states=False,
    )
    Glm5NextForConditionalGeneration.set_dflash_layers_to_capture(fake, [23, 27, 31, 35, 39, 43])
    assert fake.capture_aux_hidden_states
    assert fake.model.layers_to_capture == [24, 28, 32, 36, 40, 44]
    with pytest.raises(ValueError):
        Glm5NextForConditionalGeneration.set_dflash_layers_to_capture(fake, [45])
