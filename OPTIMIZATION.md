# PRA26 optimization report

## Scope and invariants

Target: Qwen3.5-27B BF16, one gfx936 DCU, tensor parallel size 1 and
concurrency 1. The official scheduler, `max_num_batched_tokens=4096`,
`max_num_seqs=128`, 784-token physical KV page, model weights, layer count,
attention semantics and sampling workload are unchanged. No speculative
decoding, pruning, sparse or approximate attention, skipped work, retained
low-precision weight copy, checkpoint conversion or persistent weight
quantization is used.

Every contest-specific route has exact model, platform, dtype, shape and phase
guards. Unsupported shapes and environments retain the upstream path.

## Retained optimizations

1. **Exact-shape gfx936 BF16 GEMV dispatch.** Selected measured M=1
   projections, including gate/up and full-attention QKV, use static AITER
   `gemm_a16w16` configurations. Dispatch requires gfx936, BF16, TP1, no bias,
   equal dtype/device and contiguous tensors. The packed 16480-wide GDN
   projection deliberately remains one hipBLAS GEMM because every measured
   split/AITER variant was slower. Weights remain BF16 in their original
   layout.
2. **Exact 784-page gather plus dense ROCm FlashAttention.** The normal decode
   path and a prompt's first scheduler chunk remain on paged `TRITON_ATTN`.
   For each continuation chunk of a single Qwen3.5 BF16 prefill request, a
   page-major Triton kernel gathers the exact interleaved
   `[block, 2, 784, 4, 256]` K/V values into one builder-owned dense scratch
   buffer, followed by the fixed environment's variable-length ROCm
   FlashAttention. The scratch is 128 MiB at the 32K limit, shared by all full
   attention layers and used only during the synchronous prefill call. It is
   neither a persistent KV representation nor one allocation per layer.
   Keeping the first chunk on the original reduction order minimizes accuracy
   risk for only about 39.5 ms per request in the measured model. Decode,
   mixed batches, capture, cascade, cross-attention, sliding-window, ALiBi,
   sinks and every other geometry fall back unchanged.
3. **Qwen3.5 GDN prefill fusion.** Projection splitting, convolution input
   preparation, recurrent-state packing, gated RMSNorm and post-convolution
   work are specialized for the official single-request 4096-token chunks.
   The recurrent calculation and state update are exact; scheduler padding
   rows are explicitly cleared. A strided `z` view is consumed directly to
   avoid materializing a roughly 48 MiB contiguous copy per GDN layer and
   chunk.
4. **Fused full-attention preparation.** Q gate, per-head Gemma RMSNorm,
   partial NeoX RoPE and K normalization are performed in a guarded Triton
   kernel. Gemma norm parameters are loaded in FP32 and incremented inside the
   kernel, which is numerically equivalent to constructing temporary
   `weight.float() + 1` tensors but removes those launches and allocations.
5. **Direct output projection into the layer buffer.** Both GDN and full
   attention normally materialize a `[tokens, 5120]` GEMM result and then copy
   it into a caller-owned buffer. An exact mutable custom op writes the
   unquantized TP1 BF16 `[tokens,6144] @ [6144,5120]` result directly into that
   buffer. The compilation pass explicitly defunctionalizes this op so AOT
   functionalization cannot recreate the temporary and copy. Quantization,
   bias, LoRA/batch-invariant execution, other shapes and other models retain
   the original `RowParallelLinear` path.
6. **Measured unified-attention fallback tuning.** The original paged Triton
   implementation remains the fallback and uses the measured gfx936 Qwen3.5
   M=64 query geometry only for its exact guarded shape. Logical and physical
   KV traversal and the 784-token page remain unchanged.

## Validation and screening evidence

The last public evaluation of the parent
`pra26-unified-adaptive-gdn-memory` was AC with no SLA or accuracy deduction:
19.24 / 17.04 / 13.29 tok/s for 4-8K / 8-16K / 16-32K.

The dense-attention candidate was screened with the official serving command,
two warmups, concurrency 1 and the same first five prompts. All 15 requests
completed with no errors. Paired mean TTFT changed as follows:

| Bucket | Parent TTFT | Candidate TTFT | Reduction |
|---|---:|---:|---:|
| 4-8K | 1.540 s | 1.481 s | 3.8% |
| 8-16K | 3.627 s | 3.166 s | 12.7% |
| 16-32K | 6.385 s | 5.224 s | 18.2% |

On exact synthetic tensors, the 784-page gather reproduced K/V with zero
maximum absolute error. Gather plus dense FlashAttention took approximately
4.48 / 10.01 / 22.29 ms at 8K / 16K / 32K, versus 12.67 / 28.68 / 60.73 ms
for the retained paged Triton kernel. Attention output differences caused by
the changed BF16 reduction order were at most 0.001953 / 0.000976 / 0.000488
for continuation chunks. These are screening results, not a claimed hidden-set
score.

## Rejected experiments

- Changing the physical KV page: rejected because page geometry is a locked
  contest invariant. The formal branch remains at 784.
- Splitting the 16480-wide GDN decode projection into separate matrices: all
  measured AITER/PyTorch combinations were slower than the fused hipBLAS GEMM.
- Alternate packed recurrent decode tiles: the best microbenchmark saved only
  about 4% inside a roughly 0.10 ms kernel and introduced small recurrent-state
  differences, so it was not retained.
- GDN `opt_vk`: slower at 4096 tokens and changed recurrent state/output.
- Batch-1-only compilation: engine initialization failed outside the singleton
  range. The proven dynamic plus piecewise graph configuration is retained.
- gfx936 `wvSplitK`: failed numerical validation.
- Full `ROCM_ATTN`: improved prefill but substantially regressed decode TPOT.
- Larger unified-attention tiles and eight-warps variants: slower in measured
  long-context runs and/or changed generations.
- Global hipBLASLt preference and AITER for down/out/BA/lm_head: measured
  regressions; lm_head also failed the numerical gate.

## Source locations

- `vllm/model_executor/layers/utils.py`: exact GEMV dispatch.
- `vllm/model_executor/layers/fla/ops/`: guarded GDN fusion and recurrent ops.
- `vllm/model_executor/layers/fused_qk_norm_rope.py`: Q/K norm, gate and RoPE.
- `vllm/model_executor/models/qwen3_5.py` and `qwen3_next.py`: model routing.
- `vllm/v1/attention/backends/triton_attn.py`: phase-specific attention route
  and builder-owned dense scratch.
- `vllm/v1/attention/ops/triton_qwen35_kv_gather.py`: exact paged K/V gather.
- `vllm/v1/attention/ops/triton_unified_attention.py`: guarded fallback tile.
- `vllm/platforms/rocm.py`: isolated gfx936 predicate.
