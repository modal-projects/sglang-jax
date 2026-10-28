"""DFlash draft-KV materialization into an all-SWA hybrid pool (GLM-5.3 DFlash2 draft), DP4.

034f2 (v6e-16) crashed here: ``'SWAKVPool' object has no attribute 'start_layer'``.
Run with XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, SWAKVPool
from sgl_jax.srt.speculative.dflash_worker import DFlashWorker

DP, TP, HEADS, HD, LAYERS, PAGE = 4, 2, 2, 128, 3, 8
SIZE = 64 * DP  # tokens (global); per rank 64


class _FakeDraft(nnx.Module):
    """materialize_kv like DFlashDraftModel: one (k, v) per draft layer, [N, heads, head_dim]."""

    def __init__(self):
        self.scale = nnx.Param(jnp.arange(1, LAYERS + 1, dtype=jnp.float32))

    def materialize_kv(self, target_hidden, positions):
        base = target_hidden[:, : HEADS * HD].reshape(-1, HEADS, HD)
        base = base + positions[:, None, None].astype(base.dtype)
        spec = jax.sharding.PartitionSpec("data", "tensor", None)
        out = []
        for i in range(LAYERS):
            k = (base * self.scale[i]).astype(jnp.bfloat16)
            v = (-base * self.scale[i]).astype(jnp.bfloat16)
            out.append((jax.sharding.reshard(k, spec), jax.sharding.reshard(v, spec)))
        return out


def _pool(mesh):
    pool = SWAKVPool(
        size=SIZE,
        size_swa=SIZE,
        page_size=PAGE,
        swa_attention_layer_ids=list(range(LAYERS)),
        full_attention_layer_ids=[],
        token_to_kv_pool_class=MHATokenToKVPool,
        dtype=jnp.bfloat16,
        head_num=HEADS,
        head_dim=HD,
        mesh=mesh,
        dp_size=DP,
    )
    alloc = SWATokenToKVPoolAllocator(
        size=SIZE, size_swa=SIZE, page_size=PAGE, kvcache=pool, dp_size=DP
    )
    return pool, alloc


@pytest.fixture(scope="module")
def mesh():
    if len(jax.devices()) < DP * TP:
        pytest.skip("needs 8 devices")
    return Mesh(
        np.asarray(jax.devices()[: DP * TP]).reshape(DP, TP),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


def test_materialize_into_swa_pool_dp4(mesh):
    pool, alloc = _pool(mesh)
    ref_pool, ref_alloc = _pool(mesh)
    rows_per_rank = 16
    live_per_rank = [5, 16, 0, 9]
    cache_loc = np.full(DP * rows_per_rank, -1, np.int32)
    for r, n in enumerate(live_per_rank):
        if n:
            need = -(-n // PAGE) * PAGE
            loc = alloc.alloc(need, dp_rank=r)
            ref_loc = ref_alloc.alloc(need, dp_rank=r)
            np.testing.assert_array_equal(loc, ref_loc)
            cache_loc[r * rows_per_rank : r * rows_per_rank + n] = loc[:n]
    hidden = jax.random.normal(jax.random.key(0), (DP * rows_per_rank, HEADS * HD + 7), jnp.float32)
    positions = np.arange(DP * rows_per_rank, dtype=np.int32)

    model = _FakeDraft()
    model_def, state = nnx.split(model)
    leaves, state_def = jax.tree_util.tree_flatten(state)
    runner = SimpleNamespace(
        token_to_kv_pool=pool,
        _model_def=model_def,
        _model_state_def=state_def,
        model_state_leaves=leaves,
        attn_backend=None,
    )
    worker = object.__new__(DFlashWorker)
    object.__setattr__(worker, "_worker", SimpleNamespace(model_runner=runner))
    with jax.set_mesh(mesh):
        worker._init_jit_kv_materialize()
        worker._run_jit_draft_extend(hidden, positions, cache_loc)

        # Reference: host-side full -> SWA remap, then the SWA sub-pool's own writer.
        kv = model.materialize_kv(jnp.asarray(hidden), jnp.asarray(positions))
        live = cache_loc >= 0
        for layer in range(LAYERS):
            swa_loc = np.full_like(cache_loc, -1)
            for r in range(DP):
                sl = slice(r * rows_per_rank, (r + 1) * rows_per_rank)
                m = np.asarray(ref_alloc.full_to_swa_index_mapping[r])
                swa_loc[sl] = np.where(live[sl], m[np.maximum(cache_loc[sl], 0)], -1)
            k, v = kv[layer]
            ref_pool.swa_kv_pool.set_kv_buffer(layer, jnp.asarray(swa_loc), k, v)

    for layer in range(LAYERS):
        got = np.asarray(pool.get_fused_kv_buffer(layer).astype(jnp.float32))
        want = np.asarray(ref_pool.get_fused_kv_buffer(layer).astype(jnp.float32))
        assert np.abs(want).sum() > 0
        np.testing.assert_array_equal(got, want)
