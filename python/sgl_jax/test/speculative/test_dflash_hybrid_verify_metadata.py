"""DFlash target verify on a hybrid (KDA + DSA/MLA) target with an `fa` draft (GLM-5.3 layout).

Run with 8 virtual CPU devices: XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    HybridLinearAttnBackendMetadata,
    LinearRecurrentAttnBackend,
    LinearRecurrentAttnBackendMetadata,
)
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend, MLAAttentionMetadata
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.speculative.dflash_info import DFlashVerifyInput
from sgl_jax.srt.speculative.dflash_worker import DFlashWorker
from sgl_jax.srt.speculative.draft_extend_fused import _make_target_verify_metadata

DP, TP, PER_DP_BS, T, PAGE = 4, 2, 2, 4, 16


def _bare(cls, **attrs):
    obj = object.__new__(cls)
    for k, v in attrs.items():
        object.__setattr__(obj, k, v)
    return obj


@pytest.fixture(scope="module")
def mesh():
    if len(jax.devices()) < DP * TP:
        pytest.skip("needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    return jax.sharding.Mesh(
        np.asarray(jax.devices()[: DP * TP]).reshape(DP, TP),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


def _setup(mesh):
    bs = DP * PER_DP_BS
    sh = NamedSharding(mesh, P("data"))
    linear = LinearRecurrentAttnBackend(mesh=mesh)
    mla = _bare(MLAAttentionBackend, mesh=mesh, page_size=PAGE)
    target = _bare(HybridLinearAttnBackend, full_attn_backend=mla, linear_attn_backend=linear)
    draft_backend = _bare(FlashAttention, mesh=mesh, page_size=PAGE)
    worker = _bare(
        DFlashWorker,
        block_size=T,
        mesh=mesh,
        _verify_bucket_templates={},
        _target_worker=SimpleNamespace(model_runner=SimpleNamespace(attn_backend=target)),
        _worker=SimpleNamespace(model_runner=SimpleNamespace(attn_backend=draft_backend)),
    )
    # rank 1 has one live request, the rest two.
    selector = np.array([0, 1, 2, 4, 5, 6, 7], dtype=np.int32)
    prefix = np.array([20, 3, 40, 0, 7, 9, 33, 1], dtype=np.int32)
    mwb = SimpleNamespace(
        dp_size=DP,
        per_dp_bs_size=PER_DP_BS,
        real_bs=len(selector),
        logits_indices_selector=selector,
        forward_mode=ForwardMode.TARGET_VERIFY,
        input_ids=np.ones(bs * T, dtype=np.int32),
        seq_lens=prefix,
        spec_info_padded=DFlashVerifyInput(draft_token=None, draft_token_num=T),
        recurrent_indices=np.array([1, 2, 1, 0, 1, 2, 1, 2], dtype=np.int32),
        has_initial_state=np.ones(bs, dtype=np.bool_),
        recurrent_track_indices=None,
        recurrent_track_mask=None,
    )
    template = worker._get_verify_bucket_template(mwb, bs)
    seq = prefix + template.extend_seq_lens
    aligned = (seq + PAGE - 1) // PAGE * PAGE
    cu_kv = np.zeros((DP, PER_DP_BS + 1), np.int32)
    cu_kv[:, 1:] = np.cumsum(aligned.reshape(DP, PER_DP_BS), axis=1)
    pages = np.arange(DP * 8, dtype=np.int32)
    draft_md = SimpleNamespace(
        cu_q_lens=template.cu_q_lens,
        cu_kv_lens=jax.device_put(cu_kv.reshape(-1), sh),
        page_indices=jax.device_put(pages, sh),
        seq_lens=jax.device_put(seq, sh),
        distribution=template.distribution,
    )
    return worker, mwb, draft_md, template, prefix


def test_hybrid_target_gets_hybrid_metadata(mesh):
    worker, mwb, draft_md, template, _ = _setup(mesh)
    md = worker._target_verify_forward_metadata(mwb, draft_md, template)

    assert isinstance(md, HybridLinearAttnBackendMetadata)
    full, lin = md.full_attn_metadata, md.linear_attn_metadata
    assert isinstance(full, MLAAttentionMetadata)
    assert isinstance(lin, LinearRecurrentAttnBackendMetadata)
    # Same rows / pages as the draft block.
    assert full.page_indices is draft_md.page_indices
    assert full.cu_kv_lens is draft_md.cu_kv_lens
    # MLA distribution: every live row is a multi-token ("mixed") row.
    np.testing.assert_array_equal(np.asarray(full.distribution), [0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 2])
    # KDA: block-aligned per-DP cu_q_lens, the batch's recurrent slots.
    np.testing.assert_array_equal(np.asarray(lin.cu_q_lens), np.tile([0, T, 2 * T], DP))
    np.testing.assert_array_equal(np.asarray(lin.recurrent_indices), mwb.recurrent_indices)
    # The draft's FA metadata is what the hybrid setter used to receive: it has no halves.
    assert not hasattr(draft_md, "full_attn_metadata")


def test_in_jit_relay_rebuild_keeps_hybrid_layout(mesh):
    worker, mwb, draft_md, template, prefix = _setup(mesh)
    md = worker._target_verify_forward_metadata(mwb, draft_md, template)
    sh = NamedSharding(mesh, P("data"))
    live = np.zeros(DP * PER_DP_BS, bool)
    live[mwb.logits_indices_selector] = True
    verify_lens = jax.device_put(np.where(live, prefix, 0).astype(np.int32), sh)
    alloc = jax.device_put((prefix + 2 * T).astype(np.int32), sh)

    @jax.jit
    def rebuild(md, verify_lens, alloc):
        return _make_target_verify_metadata(
            md, verify_lens, alloc, speculative_num_draft_tokens=T, page_size=PAGE, dp_size=DP
        )

    with jax.set_mesh(mesh):
        out = rebuild(md, verify_lens, alloc)
    assert isinstance(out, HybridLinearAttnBackendMetadata)
    assert isinstance(out.full_attn_metadata, MLAAttentionMetadata)
    np.testing.assert_array_equal(
        np.asarray(out.full_attn_metadata.seq_lens), np.where(live, prefix + T, 0)
    )
    np.testing.assert_array_equal(
        np.asarray(out.full_attn_metadata.distribution).reshape(DP, 3)[:, 2], [2, 1, 2, 2]
    )
    np.testing.assert_array_equal(
        np.asarray(out.linear_attn_metadata.cu_q_lens), np.asarray(md.linear_attn_metadata.cu_q_lens)
    )


def test_same_backend_target_reuses_draft_metadata(mesh):
    worker, mwb, draft_md, template, _ = _setup(mesh)
    worker.target_worker.model_runner.attn_backend = worker._worker.model_runner.attn_backend
    assert worker._target_verify_forward_metadata(mwb, draft_md, template) is draft_md


def test_dflash_draft_cell_size_matches_v16_pool(tmp_path):
    """v16 034f2 (TP16/DP4, attention TP 4): draft pool cell was 6144 bytes/token per chip."""
    import json

    from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import ModelRunnerKVCacheMixin
    from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

    (tmp_path / "config.json").write_text(
        json.dumps(
            {"num_hidden_layers": 6, "num_key_value_heads": 8, "num_attention_heads": 32,
             "head_dim": 128, "hidden_size": 4096}
        )
    )
    runner = SimpleNamespace(
        spec_algorithm=SpeculativeAlgorithm.from_string("DFLASH"),
        server_args=SimpleNamespace(speculative_draft_model_path=str(tmp_path)),
        attention_tp_size=4,
        kv_cache_dtype=jnp.bfloat16,
    )
    assert ModelRunnerKVCacheMixin._dflash_draft_cell_size(runner) == 6144
    runner.spec_algorithm = SpeculativeAlgorithm.from_string("NEXTN")
    assert ModelRunnerKVCacheMixin._dflash_draft_cell_size(runner) == 0
