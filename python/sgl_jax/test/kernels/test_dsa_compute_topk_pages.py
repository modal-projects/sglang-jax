"""compute_topk_pages bitmap fast path (pages_per_seq <= k_pages_max) == the sort/top_k reference."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa import sparse_mla


@pytest.mark.parametrize("t,sel,pps,kmax", [(1, 2048, 128, 511), (128, 2048, 128, 511), (7, 64, 32, 40), (5, 300, 600, 64)])
def test_compute_topk_pages_fast_path_matches_reference(monkeypatch, t, sel, pps, kmax):
    rng = np.random.default_rng(t * 1000 + pps)
    page = 128
    idx = rng.integers(0, pps * page + 50, size=(t, sel))  # some out-of-range pages
    idx[rng.random((t, sel)) < 0.2] = -1  # indexer padding
    idx = jnp.asarray(idx, jnp.int32)
    fast = sparse_mla.compute_topk_pages(idx, page_size=page, pages_per_seq=pps, k_pages_max=kmax)
    monkeypatch.setattr(sparse_mla, "_BITMAP_MAX_ELEMS", -1)  # force the sort/top_k path
    sparse_mla.compute_topk_pages.clear_cache()
    ref = sparse_mla.compute_topk_pages(idx, page_size=page, pages_per_seq=pps, k_pages_max=kmax)
    assert fast.shape == ref.shape == (t, kmax)
    sparse_mla.compute_topk_pages.clear_cache()
    np.testing.assert_array_equal(np.asarray(fast), np.asarray(ref))


@pytest.mark.parametrize("t,k,pps", [(1, 511, 128), (128, 511, 128), (9, 64, 16)])
def test_seq_page_lookup_matches_gather(t, k, pps):
    rng = np.random.default_rng(t + k)
    n_pages = t * pps + 7
    page_indices = jnp.asarray(rng.permutation(n_pages), jnp.int32)
    starts = jnp.asarray(np.sort(rng.integers(0, n_pages - pps, size=t)), jnp.int32)
    local = jnp.asarray(rng.integers(0, pps, size=(t, k)), jnp.int32)
    got = sparse_mla._seq_page_lookup(page_indices, starts, local, pps)
    want = page_indices[starts[:, None] + local]
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
