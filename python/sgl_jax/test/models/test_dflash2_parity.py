"""CPU parity: upstream SGLang DFlash2 draft math (PyTorch) vs the sglang-jax port.

Both sides run on the SAME real checkpoint weights (``DFLASH2_CKPT``, default the GLM-5.3-Flash
DFlash2 draft) with random target-tap inputs. The JAX side is the production module tree
(``DFlash2DraftModel`` loaded through ``WeightLoader``); only the paged-attention kernel is
replaced by a dense reference backend (`_DenseDraftAttention`), since the ragged TPU kernels do
not run on CPU. The PyTorch side is a vendored, minimal transcription of upstream
``python/sglang/srt/models/dflash.py`` (sgl-project/sglang@911693e3f816) forward math:
``DFlashDraftModel.forward`` / ``DFlashDecoderLayer`` / ``DFlashGroupedConv`` /
``_grouped_conv`` / ``CandidateSelector`` (``_score_edges``, greedy ``sample_path`` +
``_follow_maps``) / ``candidate_topk``, plus the worker's context-KV materialization
(``kv_proj_only`` -> ``apply_k_norm`` -> ``apply_k_rope`` on ``project_target_hidden``).

Env:
  DFLASH2_CKPT            checkpoint dir (config.json + model.safetensors); test skips if absent
  DFLASH2_PARITY_LAYERS   number of draft layers to run (default: all, i.e. 6)
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

_DEFAULT_CKPT = (
    "/tmp/claude-1000/-home-ec2-user/1f9a8411-d845-4cbb-b04d-9e1906e82863/scratchpad/draft"
)
CKPT = os.environ.get("DFLASH2_CKPT", _DEFAULT_CKPT)
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CKPT, "model.safetensors")),
    reason=f"DFlash2 checkpoint not found at {CKPT}",
)

CTX_LENS = (13, 21)  # two requests with different committed-context lengths
SEED = 0


# ----------------------------------------------------------------------------------------------
# Vendored upstream reference math (fp32).
# ----------------------------------------------------------------------------------------------


# When set, the reference rounds every GEMM / norm output to bf16 (fp32 statistics), i.e.
# what upstream SGLang computes in its production dtype. Used to measure the bf16 noise floor.
_EMULATE_BF16 = False


def _ref_rms(x, w, eps):
    x = x.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if _EMULATE_BF16:
        return (y.to(torch.bfloat16) * w.to(torch.bfloat16)).float()
    return y * w.float()


def _ref_linear(x, w):
    if _EMULATE_BF16:
        return F.linear(x.to(torch.bfloat16), w.to(torch.bfloat16)).float()
    return F.linear(x.float(), w.float())


def _ref_rope(x, positions, theta):
    # neox-style rotary over the full head_dim (upstream get_rope, is_neox_style=True)
    d = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    cos, sin = freqs.cos()[:, None, :], freqs.sin()[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _ref_grouped_conv(hidden_states, delta, base, block_size, num_groups, group_size, taps):
    # verbatim upstream _grouped_conv
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    out = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        shifted = torch.where((position >= tap).view(-1, 1, 1), shifted, 0)
        out = out + coefficients[:, tap] * shifted
    return out.flatten(-2)


class _RefConv:
    def __init__(self, w, prefix, hidden, block_size, taps, group_size):
        self.base = w[f"{prefix}.base_kernel"].float()
        self.proj = w[f"{prefix}.kernel_projection.weight"]
        self.block_size, self.taps, self.group_size = block_size, taps, group_size
        self.num_groups = hidden // group_size

    def _convolve(self, x, delta, side):
        return _ref_grouped_conv(
            x, delta, self.base[side], self.block_size, self.num_groups, self.group_size, self.taps
        )

    def prepare(self, x):
        c = _ref_linear(x, self.proj).reshape(*x.shape[:-1], 2, self.taps, self.num_groups)
        return self._convolve(x, c[..., 0, :, :], 0), c[..., 1, :, :]

    def finish(self, x, c):
        return self._convolve(x, c, 1)


def _ref_dense_attention(q, k, v, q_pos, k_pos, causal, window, scaling):
    """q [Tq, Hq, D], k/v [Tk, Hkv, D]; GQA; mask by absolute positions."""
    rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    s = torch.einsum("qhd,khd->hqk", q, k) * scaling
    mask = torch.ones(q.shape[0], k.shape[0], dtype=torch.bool)
    if causal:
        mask &= q_pos[:, None] >= k_pos[None, :]
    if window:
        mask &= (q_pos[:, None] - k_pos[None, :]) < window
    s = s.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", s.softmax(-1), v)


class TorchDFlash2Reference:
    def __init__(self, weights, cfg, num_layers):
        self.w, self.cfg, self.num_layers = weights, cfg, num_layers
        d = cfg["dflash_config"]
        self.block_size = d["block_size"]
        self.eps = cfg["rms_norm_eps"]
        self.hd = cfg["head_dim"]
        self.nh, self.nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.window = cfg["sliding_window"]
        self.causal = bool(cfg.get("is_causal", True))
        self.top_k = d["selector_top_k"]
        self.output_multiplier = float(d.get("output_multiplier", 1.0))
        H = cfg["hidden_size"]
        self.convs = [
            (
                _RefConv(
                    weights,
                    f"layers.{i}.attention_conv",
                    H,
                    self.block_size,
                    d["conv_kernel_size"],
                    d["conv_group_size"],
                ),
                _RefConv(
                    weights,
                    f"layers.{i}.mlp_conv",
                    H,
                    self.block_size,
                    d["conv_kernel_size"],
                    d["conv_group_size"],
                ),
            )
            for i in range(num_layers)
        ]

    def project_target_hidden(self, target_hidden):
        return _ref_rms(
            _ref_linear(target_hidden, self.w["fc.weight"]), self.w["hidden_norm.weight"], self.eps
        )

    def ctx_kv(self, i, ctx_hidden, positions):
        p = f"layers.{i}.self_attn"
        k = _ref_linear(ctx_hidden, self.w[f"{p}.k_proj.weight"]).view(-1, self.nkv, self.hd)
        v = _ref_linear(ctx_hidden, self.w[f"{p}.v_proj.weight"]).view(-1, self.nkv, self.hd)
        k = _ref_rms(k, self.w[f"{p}.k_norm.weight"], self.eps)
        return _ref_rope(k, positions, self.theta), v

    def attention(self, i, x, positions, ctx, ctx_pos, block_pos):
        p = f"layers.{i}.self_attn"
        q = _ref_linear(x, self.w[f"{p}.q_proj.weight"]).view(-1, self.nh, self.hd)
        k = _ref_linear(x, self.w[f"{p}.k_proj.weight"]).view(-1, self.nkv, self.hd)
        v = _ref_linear(x, self.w[f"{p}.v_proj.weight"]).view(-1, self.nkv, self.hd)
        q = _ref_rope(_ref_rms(q, self.w[f"{p}.q_norm.weight"], self.eps), positions, self.theta)
        k = _ref_rope(_ref_rms(k, self.w[f"{p}.k_norm.weight"], self.eps), positions, self.theta)
        outs = []
        bs = self.block_size
        for r, (ck, cv) in enumerate(ctx[i]):
            sl = slice(r * bs, (r + 1) * bs)
            kk = torch.cat([ck, k[sl]])
            vv = torch.cat([cv, v[sl]])
            kpos = torch.cat([ctx_pos[r], block_pos[r]])
            outs.append(
                _ref_dense_attention(
                    q[sl], kk, vv, block_pos[r], kpos, self.causal, self.window, self.hd**-0.5
                )
            )
        o = torch.cat(outs).reshape(x.shape[0], -1)
        return _ref_linear(o, self.w[f"{p}.o_proj.weight"])

    def forward(self, input_embeds, positions, ctx, ctx_pos, block_pos):
        h, residual = input_embeds.float(), None
        for i in range(self.num_layers):
            p = f"layers.{i}"
            residual = h if residual is None else h + residual
            x = _ref_rms(residual, self.w[f"{p}.input_layernorm.weight"], self.eps)
            aconv, mconv = self.convs[i]
            x, akern = aconv.prepare(x)
            a = self.attention(i, x, positions, ctx, ctx_pos, block_pos)
            a = aconv.finish(a, akern)
            residual = a + residual
            x = _ref_rms(residual, self.w[f"{p}.post_attention_layernorm.weight"], self.eps)
            x, mkern = mconv.prepare(x)
            g = _ref_linear(x, self.w[f"{p}.mlp.gate_proj.weight"])
            u = _ref_linear(x, self.w[f"{p}.mlp.up_proj.weight"])
            h = _ref_linear(F.silu(g) * u, self.w[f"{p}.mlp.down_proj.weight"])
            h = mconv.finish(h, mkern)
        return _ref_rms(h + residual, self.w["norm.weight"], self.eps)

    # --- selector (upstream CandidateSelector + candidate_topk, greedy) ---
    def compute_candidates(self, hidden, lm_head, vocab):
        logits = torch.matmul(hidden.float(), lm_head[:vocab].float().T)
        vals, ids = torch.topk(logits, self.top_k, dim=-1)
        return ids.long(), vals.float() * self.output_multiplier

    def build_lattice(self, candidate_ids, unary_logits, hidden_states, anchor_token_ids):
        hidden = _ref_linear(hidden_states, self.w["candidate_selector.hidden_projection.weight"])
        pred_t = self.w["candidate_selector.predecessor_codebook"].float()
        succ_t = self.w["candidate_selector.successor_codebook"].float()
        keys = succ_t[candidate_ids]
        predecessor_ids = torch.cat(
            [anchor_token_ids[:, None, None].expand(-1, 1, self.top_k), candidate_ids[:, :-1]],
            dim=1,
        )
        predecessors = pred_t[predecessor_ids]
        return unary_logits[:, :, None] + torch.einsum(
            "blpr,blcr->blpc", predecessors * hidden[:, :, None], keys
        )

    @staticmethod
    def greedy_path(candidate_ids, scores):
        initial = scores[:, 0, 0].argmax(dim=-1)
        local_maps = scores[:, 1:].argmax(dim=-1)
        index, path = initial, [initial]
        for edge in range(scores.shape[1] - 1):
            index = local_maps[:, edge].gather(-1, index[:, None])[:, 0]
            path.append(index)
        path = torch.stack(path, dim=1)
        return candidate_ids.gather(-1, path.unsqueeze(-1))[:, :, 0]


# ----------------------------------------------------------------------------------------------
# JAX side.
# ----------------------------------------------------------------------------------------------


class _DenseDraftAttention:
    """Stands in for the paged attention backend: block queries over [ctx KV ++ block KV].

    Honors the layer's attn_type (causal within the block) and sliding window, exactly the
    semantics the ragged kernels implement for TARGET_VERIFY-shaped draft blocks.
    """

    def __init__(self, ctx_kv, ctx_pos, block_pos, block_size):
        self.ctx_kv, self.ctx_pos, self.block_pos, self.bs = ctx_kv, ctx_pos, block_pos, block_size

    def __call__(self, q, k, v, layer, forward_batch, token_to_kv_pool, **kwargs):
        import jax
        import jax.numpy as jnp
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.layers.radix_attention import AttentionType

        rep3 = P(None, None, None)
        q, k, v = (jax.sharding.reshard(x, rep3) for x in (q, k, v))
        causal = layer.attn_type == AttentionType.DECODER
        window = layer.sliding_window_size
        outs = []
        for r, (ck, cv) in enumerate(self.ctx_kv[layer.layer_id]):
            sl = slice(r * self.bs, (r + 1) * self.bs)
            ck, cv = jax.sharding.reshard(ck, rep3), jax.sharding.reshard(cv, rep3)
            kk = jnp.concatenate([ck, k[sl]]).astype(jnp.float32)
            vv = jnp.concatenate([cv, v[sl]]).astype(jnp.float32)
            qq = q[sl].astype(jnp.float32)
            qpos = self.block_pos[r]
            kpos = jnp.concatenate([self.ctx_pos[r], self.block_pos[r]])
            rep = qq.shape[1] // kk.shape[1]
            kk = jnp.broadcast_to(kk[:, :, None], (kk.shape[0], kk.shape[1], rep, kk.shape[2]))
            vv = jnp.broadcast_to(vv[:, :, None], (vv.shape[0], vv.shape[1], rep, vv.shape[2]))
            kk = kk.reshape(kk.shape[0], -1, kk.shape[-1])
            vv = vv.reshape(vv.shape[0], -1, vv.shape[-1])
            s = jnp.einsum("qhd,khd->hqk", qq, kk) * layer.scaling
            mask = jnp.ones((qq.shape[0], kk.shape[0]), dtype=bool)
            if causal:
                mask &= qpos[:, None] >= kpos[None, :]
            if window:
                mask &= (qpos[:, None] - kpos[None, :]) < window
            s = jnp.where(mask[None], s, -jnp.inf)
            p = jax_softmax(s)
            outs.append(jnp.einsum("hqk,khd->qhd", p, vv))
        o = jnp.concatenate(outs).astype(q.dtype)
        return o.reshape(q.shape[0], -1), None


def jax_softmax(x):
    import jax

    return jax.nn.softmax(x, axis=-1)


def _load_jax_model(dtype_name, num_layers):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jax._src.mesh import AxisType
    from jax.sharding import Mesh

    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.models.registry import ModelRegistry

    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    mc = ModelConfig(model_path=CKPT, dtype=dtype_name, is_draft_model=True)
    mc.hf_config.num_hidden_layers = num_layers
    model_cls, arch = ModelRegistry.resolve_model_cls(mc.hf_config.architectures)
    assert arch == "DFlash2DraftModel" and model_cls.__name__ == "DFlash2DraftModel"
    dtype = jnp.float32 if dtype_name == "float32" else jnp.bfloat16
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(lambda: model_cls(mc.hf_config, mesh=mesh, dtype=dtype))
        model.load_weights(mc)
    return model, mesh


def _run_jax(model, mesh, inputs, ref_cfg):
    import jax
    import jax.numpy as jnp

    bs = ref_cfg["dflash_config"]["block_size"]
    dt = model.dtype
    with jax.set_mesh(mesh):
        ctx_hidden_all = []
        ctx_kv = {}
        for r, th in enumerate(inputs["target_hidden"]):
            kv = model.materialize_kv(jnp.asarray(th, dtype=dt), jnp.asarray(inputs["ctx_pos"][r]))
            for i, (k, v) in enumerate(kv):
                ctx_kv.setdefault(i, []).append((k, v))
            ctx_hidden_all.append(kv)
        backend = _DenseDraftAttention(
            ctx_kv,
            [jnp.asarray(p) for p in inputs["ctx_pos"]],
            [jnp.asarray(p) for p in inputs["block_pos"]],
            bs,
        )
        fb = SimpleNamespace(
            attn_backend=backend,
            positions=jnp.asarray(inputs["positions"], dtype=jnp.int32),
            input_embedding=jnp.asarray(inputs["input_embeds"], dtype=dt),
        )
        out, _, _, _ = model(fb, SimpleNamespace(token_to_kv_pool=None), None)
        hidden = out.hidden_states.reshape(len(CTX_LENS), bs, -1)
        tokens, ids, unary, scores = model.propose_block_tokens(
            hidden[:, 1:, :],
            jnp.asarray(inputs["lm_head"], dtype=dt),
            jnp.asarray(inputs["anchors"], dtype=jnp.int32),
            vocab_size=inputs["vocab"],
            return_lattice=True,
        )
    f32 = lambda a: np.asarray(jnp.asarray(a, dtype=jnp.float32))
    return dict(
        hidden=f32(hidden),
        tokens=np.asarray(tokens),
        ids=np.asarray(ids),
        unary=f32(unary),
        scores=f32(scores),
    )


# ----------------------------------------------------------------------------------------------


def _err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    abs_err = np.abs(a - b)
    return (
        float(abs_err.max()),
        float(abs_err.max() / (np.abs(b).max() + 1e-12)),
        float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)),
    )


@pytest.fixture(scope="module")
def reference():
    return build_reference()


def build_reference(emulate_bf16: bool = False):
    global _EMULATE_BF16
    _EMULATE_BF16 = emulate_bf16
    try:
        return _build_reference()
    finally:
        _EMULATE_BF16 = False


def _build_reference():
    from safetensors.torch import load_file

    with open(os.path.join(CKPT, "config.json")) as f:
        cfg = json.load(f)
    num_layers = int(os.environ.get("DFLASH2_PARITY_LAYERS", cfg["num_hidden_layers"]))
    weights = load_file(os.path.join(CKPT, "model.safetensors"))
    weights = {
        k: v
        for k, v in weights.items()
        if not (k.startswith("layers.") and int(k.split(".")[1]) >= num_layers)
    }
    d = cfg["dflash_config"]
    H, V = cfg["hidden_size"], cfg["vocab_size"]
    bs, n_taps = d["block_size"], len(d["target_layer_ids"])
    g = torch.Generator().manual_seed(SEED)
    # Random target taps (mHC-contracted residual of each tapped layer), anchors, embeddings
    # and a stand-in target LM head; all stored bf16 so both sides see identical values.
    target_hidden = [torch.randn(n, n_taps * H, generator=g).to(torch.bfloat16) for n in CTX_LENS]
    anchors = torch.randint(0, V, (len(CTX_LENS),), generator=g)
    embed_anchor = (0.02 * torch.randn(len(CTX_LENS), H, generator=g)).to(torch.bfloat16)
    embed_mask = (0.02 * torch.randn(H, generator=g)).to(torch.bfloat16)
    lm_head = (0.02 * torch.randn(V, H, generator=g)).to(torch.bfloat16)
    input_embeds = torch.cat(
        [
            torch.cat([embed_anchor[r : r + 1], embed_mask[None].expand(bs - 1, H)])
            for r in range(len(CTX_LENS))
        ]
    )
    ctx_pos = [torch.arange(n) for n in CTX_LENS]
    block_pos = [torch.arange(n, n + bs) for n in CTX_LENS]
    positions = torch.cat(block_pos)

    ref = TorchDFlash2Reference(weights, cfg, num_layers)
    with torch.no_grad():
        ctx = {}
        for r, th in enumerate(target_hidden):
            ch = ref.project_target_hidden(th)
            for i in range(num_layers):
                ctx.setdefault(i, []).append(ref.ctx_kv(i, ch, ctx_pos[r]))
        hidden = ref.forward(input_embeds, positions, ctx, ctx_pos, block_pos)
        hidden = hidden.view(len(CTX_LENS), bs, H)
        pred = hidden[:, 1:, :]
        ids, unary = ref.compute_candidates(pred.reshape(-1, H), lm_head, V)
        ids = ids.view(len(CTX_LENS), bs - 1, -1)
        unary = unary.view(len(CTX_LENS), bs - 1, -1)
        scores = ref.build_lattice(ids, unary, pred, anchors)
        tokens = ref.greedy_path(ids, scores)
    del weights, ref
    inputs = dict(
        target_hidden=[t.float().numpy() for t in target_hidden],
        anchors=anchors.numpy().astype(np.int32),
        input_embeds=input_embeds.float().numpy(),
        lm_head=lm_head.float().numpy(),
        ctx_pos=[p.numpy().astype(np.int32) for p in ctx_pos],
        block_pos=[p.numpy().astype(np.int32) for p in block_pos],
        positions=positions.numpy().astype(np.int32),
        vocab=V,
    )
    out = dict(
        hidden=hidden.numpy(),
        tokens=tokens.numpy(),
        ids=ids.numpy(),
        unary=unary.numpy(),
        scores=scores.numpy(),
    )
    return cfg, num_layers, inputs, out


def _compare(tag, jx, rf):
    h = _err(jx["hidden"], rf["hidden"])
    u = _err(jx["unary"], rf["unary"])
    s = _err(jx["scores"], rf["scores"])
    id_match = float((jx["ids"] == rf["ids"]).mean())
    tok_match = float((jx["tokens"] == rf["tokens"]).mean())
    print(
        f"\n[{tag}] draft hidden  max_abs={h[0]:.3e} max_rel={h[1]:.3e} rel_fro={h[2]:.3e}"
        f"\n[{tag}] unary top-k   max_abs={u[0]:.3e} max_rel={u[1]:.3e} rel_fro={u[2]:.3e}"
        f"\n[{tag}] lattice       max_abs={s[0]:.3e} max_rel={s[1]:.3e} rel_fro={s[2]:.3e}"
        f"\n[{tag}] top-k id match={id_match:.4f}  draft-token match={tok_match:.4f}"
        f"\n[{tag}] jax tokens={jx['tokens'].tolist()}\n[{tag}] ref tokens={rf['tokens'].tolist()}"
    )
    return h, u, s, id_match, tok_match


def test_dflash2_parity_fp32(reference):
    """Same math, fp32 on both sides: tight agreement and identical draft tokens."""
    cfg, num_layers, inputs, rf = reference
    model, mesh = _load_jax_model("float32", num_layers)
    jx = _run_jax(model, mesh, inputs, cfg)
    del model
    h, u, s, id_match, tok_match = _compare(f"fp32 L={num_layers}", jx, rf)
    assert h[2] < 1e-4 and h[1] < 1e-3
    assert s[1] < 1e-3
    assert id_match == 1.0
    assert tok_match == 1.0


def test_dflash2_parity_bf16(reference):
    """Production dtype (bf16) vs the fp32 reference, judged against upstream's own bf16 noise.

    The same upstream math rounded to bf16 per op (``build_reference(emulate_bf16=True)``) sets
    the noise floor; the JAX bf16 port must stay within 2x of it. Tokens are reported, not
    asserted: with the random stand-in LM head the top-K logits are near-tied, so bf16 rounding
    alone reorders candidates (upstream-bf16 itself does).
    """
    cfg, num_layers, inputs, rf = reference
    _, _, _, rf_bf16 = build_reference(emulate_bf16=True)
    floor = _compare(f"upstream-bf16 L={num_layers}", rf_bf16, rf)
    model, mesh = _load_jax_model("bfloat16", num_layers)
    jx = _run_jax(model, mesh, inputs, cfg)
    del model
    h, u, s, id_match, tok_match = _compare(f"jax-bf16 L={num_layers}", jx, rf)
    assert h[2] < 2.0 * floor[0][2] + 1e-3
    assert u[2] < 2.0 * floor[1][2] + 1e-3
    assert s[2] < 2.0 * floor[2][2] + 1e-2
    assert np.isfinite(jx["scores"]).all()


def test_dflash2_checkpoint_coverage_is_strict(reference):
    """Every checkpoint tensor is mapped; an unknown tensor or a missing one fails loudly."""
    import jax

    cfg, num_layers, _, _ = reference
    model, _ = _load_jax_model("bfloat16", 1) if num_layers != 1 else _load_jax_model("bfloat16", 1)
    mappings = model._create_weight_mappings()
    from safetensors import safe_open

    with safe_open(os.path.join(CKPT, "model.safetensors"), "np") as f:
        keys = set(f.keys())
    model._check_checkpoint_coverage(keys, mappings)  # layers >= 1 are an explicit truncation
    with pytest.raises(ValueError, match="unmapped checkpoint tensors"):
        model._check_checkpoint_coverage(keys | {"layers.0.surprise.weight"}, mappings)
    with pytest.raises(ValueError, match="missing checkpoint tensors"):
        model._check_checkpoint_coverage(keys - {"candidate_selector.successor_codebook"}, mappings)
    full = 15 * cfg["num_hidden_layers"] + 6
    assert len(keys) == full
    del jax
