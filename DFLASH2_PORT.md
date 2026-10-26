# DFlash2 on sglang-jax (GLM-5.3-Flash draft)

Branch `glm53-dflash2` (from `glm53-v6e`). This is a port of upstream SGLang's DFlash2 draft
(`sgl-project/sglang@911693e3f816`: `python/sglang/srt/models/dflash.py`,
`speculative/dflash_utils.py`, `speculative/dflash_worker_v2.py`, `models/glm5_next.py`,
`kernels/ops/layernorm/mhc.py`) to JAX/Flax NNX. Nothing here has been run on a TPU yet. All
of the validation below ran on CPU.

## What DFlash2 changes vs DFlash v1

| | DFlash v1 (`DFlashDraftModel`) | DFlash2 (`DFlash2DraftModel`) |
|---|---|---|
| Context features | concat of the target taps -> `fc` -> `hidden_norm`, then the per-layer K/V of the committed context (`k_norm` + RoPE, no input norm, no conv) | unchanged |
| Draft block | `[verified token, mask x (B-1)]`, query = whole block | unchanged (block 8, mask id 154856) |
| Block attention | bidirectional (`ENCODER_ONLY`) | **causal, sliding window 4096**: `layer_types=sliding_attention` and `is_causal=true`, the same rule upstream applies in `_get_dflash_layer_attention_params` |
| Per-layer extra | none | **two grouped dynamic depthwise convolutions** per layer (`attention_conv`, `mlp_conv`) |
| Token proposal | argmax of `target_lm_head(h)` per slot | **candidate selector**: top-K (16) of the target head per slot, then a low-rank K x K transition lattice that is walked from the anchor |
| Own weights | layers, `fc`, `hidden_norm`, `norm` | the same, plus `layers.*.{attention,mlp}_conv.{base_kernel,kernel_projection.weight}` and `candidate_selector.{predecessor_codebook,successor_codebook,hidden_projection.weight}` (96 tensors for 6 layers, all mapped) |

**Grouped conv** (`DFlashGroupedConv`, `conv_kernel_size=2` taps, `conv_group_size=16`, so 256
groups). `kernel_projection(x)` produces `[2 sides, taps, groups]` per token. The first side
convolves the sublayer *input* (`prepare`) and the second convolves the sublayer *output*
(`finish`):
`out[n,c] = sum_t (base[side,t,c] + delta[n,t,c//16]) * x[n-t,c] * [n % B >= t]`.
The convolution is causal inside a block and never reaches across a block boundary. Each
layer runs: `norm -> conv.prepare -> attn -> conv.finish -> +res -> norm -> conv.prepare -> mlp -> conv.finish`.
Because tokens are indexed by their position inside the block, every draft row has to hold the
full block. DFlash2 therefore rejects `--speculative-sample-from-anchor`. `set_block_size`
passes the verify width on to the convs, as upstream does.

**Selector** (`CandidateSelector`, rank 256, K = 16). The draft has no LM head of its own.
For mask slots `e = 1..B-1`:
`score[b,e,p,c] = unary[b,e,c] + <A[pred[b,e,p]] * W h[b,e], B[cand[b,e,c]]>`.
Here `unary` is the top-K target logits times `output_multiplier` (with optional softcap), and
`pred` is the verified anchor for slot 0 and otherwise the previous slot's candidates. The
greedy walk takes the argmax of slot 0's anchor row and, at each later slot, the argmax of the
row picked at the slot before. This is upstream's `sample_path` with `greedy_mask=True`.
Top-K under TP runs as a local top-k per vocab shard, then an all-gather of only K values per
shard, then a global top-k (`candidate_topk`, a `shard_map`). LM-head padding columns are
masked out.

## How the target taps are formed for GLM-5.3 (mHC)

This follows upstream `Glm5NextModel`. `set_dflash_layers_to_capture(ids)` captures before
layer `id+1`, which is the finished output of layer `id`. Upstream captures
`stream.residual` in the attention boundary's `prepare`. With mHC that residual is **all
`hc_mult=4` residual streams** `[N, 4*4096]`: after layer `id`'s FFN `hc_post`, before any
norm, and before the next layer's `hc_pre` mixing. `_prepare_aux_hidden_state` then applies
`hc_contract(x, 4) = x.unflatten(-1, (4, -1)).mean(-2)`, a plain **mean over the 4 streams**
with no weights and no norm. The 6 taps (`[23,27,31,35,39,43]`) are **concatenated** in
that order into `[N, 24576]`. The draft projects them with `fc` (24576 -> 4096) followed by
`hidden_norm`. In sglang-jax, `Glm5NextModel` now does the same: `hc_contract` is taken at
`layers_to_capture`, and the `aux_hidden_states` go through the logits processor's FULL
hidden capture (`models/glm5_next.py`).

## Draft and verify in the sglang-jax DFlash worker

1. The worker picks `DFlash2DraftModel` by checkpoint architecture
   (`_resolve_draft_model_class`). It checks that the selector vocab equals the target vocab
   and that the draft query width equals the block, then calls `set_block_size`.
2. **Draft (one jit):** embed `[verified, mask x7]` with the target embedding. Run the draft
   over the block, attending causally (SWA) to the draft KV of the committed context plus the
   block. Take the hidden states at slots 1..7, then
   `propose_block_tokens(hidden, target_lm_head, anchor)`: logits, top-16, lattice, greedy
   walk. The result is 7 draft tokens.
3. **Verify (one jit):** the target scores `[seed, d1..d7]` in TARGET_VERIFY.
   `dflash_greedy_verify` gives `accept_len` and the bonus token. For hybrid targets the
   recurrent state is committed at the accepted position in the same jit (see below).
4. The target taps of the accepted tokens become draft context K/V through `materialize_kv`,
   which is unchanged from v1.

Only greedy is supported. The JAX DFlash worker is greedy-only, so the selector's sampling
walk (`q_rows` with sparse rejection sampling) is not ported.

## KDA target-verify (implemented here)

Before this branch, `KDAAttnBackend` raised on TARGET_VERIFY, and so did
`LinearRecurrentAttnBackend.get_forward_metadata`. Now:

* **Metadata:** TARGET_VERIFY produces block-aligned `cu_q_lens = arange(bs+1) * T`, with `T`
  taken from `spec_info.draft_token_num`. The token axis must be exactly `rows * T`.
* **Verify forward** (`KDAAttnBackend._forward_verify`): the short conv uses the extend path
  over each request's T tokens, followed by l2norm and the activated gate. The exact
  recurrence (`naive_recurrent_kda` over `[B, T]`) gives every position's output. **No pool
  write happens.** The layer returns a `KDAVerifyRecord` in place of its new SSM buffer. The
  record holds the pre-verify state `h0`, the conv caches, the raw pre-conv q/k/v, and the
  per-token k/v/gate/beta.
* **Commit** (`commit_kda_verify(record, n)`, where `n = accept_len + 1` is the seed plus the
  accepted drafts). It replays the first `n` tokens from `h0`. Steps past `n` are masked with
  `gate=0, beta=0`, which is an exact no-op, so shapes stay static. The conv cache becomes the
  last `K-1` columns of `conv0 ++ x[:n]`. Rows with `idx==0` are not scattered. The design
  keeps per-token *inputs* instead of checkpointing T full `[64,128,128]` fp32 states per
  request per layer, which would be 4 MB x T per request per layer. The cost is about
  0.4 MB/request/layer and a replay of at most T steps.
* **Wiring:** `commit_recurrent_verify(pool_updates, accept_lens+1)` runs inside the DFlash
  target-verify jit. It does nothing for non-hybrid targets. `_make_target_verify_metadata`
  now handles `HybridLinearAttnBackendMetadata`, rebuilding only the full-attention part.

## Test results (CPU, `JAX_PLATFORMS=cpu`)

`python/sgl_jax/test/models/test_dflash2_parity.py` uses the real GLM-5.3 DFlash2 checkpoint,
all 6 layers, loaded through the production `WeightLoader`, with strict coverage checks.
Inputs: 2 requests with context lengths 13 and 21, random bf16 target taps and embeddings,
and a random stand-in target LM head of 154880 x 4096. The torch side is a vendored
transcription of the upstream math, run in fp32. On the JAX side only the paged-attention
kernel is swapped, for a dense backend with the same causal/SWA semantics.

| comparison (6 layers) | draft hidden max_abs / max_rel / rel_fro | unary top-K rel_fro | lattice rel_fro | top-K id match | draft-token match |
|---|---|---|---|---|---|
| JAX fp32 vs upstream fp32 | 2.06e-4 / 2.01e-5 / 1.43e-5 | 4.14e-6 | 8.09e-6 | 1.000 | **1.000 (14/14)** |
| upstream math rounded to bf16 vs fp32 (noise floor) | 6.86e-1 / 6.69e-2 / 6.39e-2 | 1.07e-2 | 4.08e-1 | 0.429 | 0.500 |
| JAX bf16 vs upstream fp32 | 1.53e+0 / 1.49e-1 / 1.06e-1 | 1.32e-2 | 4.70e-1 | 0.272 | 0.214 |

With 2 layers (`DFLASH2_PARITY_LAYERS=2`), fp32 gives a hidden rel_fro of 4.1e-6 with 14/14
tokens matching, and bf16 gives 3.2e-2 against a floor of 2.4e-2. The bf16 test checks that
the JAX error stays within 2x of upstream's own bf16 error. bf16 token agreement is low only
because the random stand-in head leaves the top-16 logits nearly tied: upstream's own bf16
math matches its fp32 tokens only 50% of the time.

`test_dflash2.py` (10 tests) covers:
* registry
* attention-type rules
* rejection of unported options
* conv against NumPy, including isolation between blocks
* mapping equal to the parameter set
* `propose_block_tokens` under `jax.jit` equal to eager and to an upstream-greedy NumPy walk
* TP top-k on 4 virtual CPU devices (padding excluded)
* worker class resolution
* `set_block_size`
* GLM tap bookkeeping (`[23..43]` -> capture `[24..44]`) and `hc_contract`

`test_kda_verify.py` (4 tests, fp32 and bf16, also run with tp=4 on 4 virtual devices)
compares against T=5 sequential single-token DECODE calls through the same backend:
* verify outputs match the decode outputs exactly (max_abs 0.0)
* `commit(n)` matches the state after n decodes for every per-request n in {0..5} (ssm
  max_abs at most 3e-8, conv exact)
* verify leaves the pool untouched and slot 0 is never written

Existing suites are unchanged: `test_dflash.py`, `test_dflash_worker.py`, `test_dflash_info.py`
and `test_dflash_server_args.py` all pass. 3 eagle-tree tests and 9 `test_kda_attention.py`
tests also fail on CPU on the base branch.

## Still missing for end-to-end GLM-5.3 + DFlash2 spec in sglang-jax

1. **No TPU or end-to-end run yet.** The DFlash2 draft jit, the GLM verify jit with KDA records
   and commit, and the relay/overlap path have not been compiled or run on TPU.
   * Ragged attention with `attn_type=DECODER` and `sliding_window=4096` on the draft's
     TARGET_VERIFY-shaped metadata is assumed correct, not tested.
   * The context length the draft sees is bounded by `SGL_JAX_DFLASH_DRAFT_PREFIX_WINDOW`
     and the KV layout.
2. **Scheduler and batches for hybrid verify.** Not yet checked:
   * that DFlash verify `ModelWorkerBatch`es (including the precompile dummy batches in
     `_make_verify_dummy_batch`) carry `recurrent_indices` and `has_initial_state`;
   * that the token axis is exactly `bs * T`.

   Prefix-cache **track** slots are not written during verify. Extra-buffer / radix tracking
   of recurrent state at accepted boundaries still needs a design.
3. **DSA layers in verify.** GLM's sparse-MLA backend has a spec path (`DSA_SPEC_AS_DECODE`)
   that has not been exercised with DFlash here. `_make_target_verify_metadata` rebuilds
   metadata via `type(old)(**kwargs)`, and any DSA-specific metadata fields need checking.
4. **Performance.**
   * Exact `lax.top_k` over 154880/tp columns for every slot, where upstream uses flashinfer
     radix top-k. Try `approx_max_k`, or fusing the head GEMM with top-k.
   * The selector codebooks (2 x 154880 x 256 bf16) are replicated.
   * The `naive_recurrent_kda` scan for the verify window and the commit has no Pallas
     multi-token decode kernel.
   * FULL capture of 6 x 4096 taps for every target token, which is large at prefill.
5. **Sampling (non-greedy) DFlash2.** This needs the selector's sampled walk with `q_rows`
   and sparse rejection sampling against the target (upstream `_selector_sampling_accept`).
6. **Not ported (these raise if configured):** attention sink bias, `attention_value_scale`,
   and the Domino projector.
