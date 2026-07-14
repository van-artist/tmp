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
   split/AITER variant was slower.
2. **BF16 gate/up GEMV plus SwiGLU fusion.** Dense-layer M=1 decode uses a
   model-specific Triton kernel that accumulates both original BF16 weight
   halves and immediately applies the original SiLU-and-multiply sequence.
   The kernel explicitly reproduces both intermediate BF16 round trips, keeps
   the checkpoint weight and output in BF16, and creates no alternate weight
   representation. Every non-M=1, non-gfx936, non-BF16, noncontiguous or
   mismatched shape falls back to the original BF16 GEMV and activation.
3. **Pack the tiny GDN b/a projection into qkvz.** Qwen3.5-27B has only 96 b/a
   outputs beside 16,384 q/k/v/z outputs. Its six original BF16 checkpoint
   shards are loaded into one 16,480-wide `MergedColumnParallelLinear`, and
   the former b/a module is a parameter-free `nn.Identity`. This removes one
   launch in each GDN layer without changing any tensor value or precision.
4. **Exact 784-page gather plus dense ROCm FlashAttention.** The normal decode
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
5. **Qwen3.5 GDN prefill fusion.** Projection splitting, convolution input
   preparation, recurrent-state packing, gated RMSNorm and post-convolution
   work are specialized for the official single-request 4096-token chunks.
   The recurrent calculation and state update are exact; scheduler padding
   rows are explicitly cleared. A strided `z` view is consumed directly to
   avoid materializing a roughly 48 MiB contiguous copy per GDN layer and
   chunk.
6. **Fused full-attention preparation.** Q gate, per-head Gemma RMSNorm,
   partial NeoX RoPE and K normalization are performed in a guarded Triton
   kernel. Gemma norm parameters are loaded in FP32 and incremented inside the
   kernel, which is numerically equivalent to constructing temporary
   `weight.float() + 1` tensors but removes those launches and allocations.
7. **Direct output projection into the layer buffer.** Both GDN and full
   attention normally materialize a `[tokens, 5120]` GEMM result and then copy
   it into a caller-owned buffer. An exact mutable custom op writes the
   unquantized TP1 BF16 `[tokens,6144] @ [6144,5120]` result directly into that
   buffer. The compilation pass explicitly defunctionalizes this op so AOT
   functionalization cannot recreate the temporary and copy. Quantization,
   bias, LoRA/batch-invariant execution, other shapes and other models retain
   the original `RowParallelLinear` path.
8. **Measured unified-attention fallback tuning.** The original paged Triton
   implementation remains the fallback and uses the measured gfx936 Qwen3.5
   M=64 query geometry only for its exact guarded shape. Logical and physical
   KV traversal and the 784-token page remain unchanged.
9. **Scalar page lookup for decode tiles.** Decode uses a 16-token K/V tile,
   and the fixed 784-token physical page is exactly 49 such tiles. Therefore a
   tile cannot cross a page boundary: the 3-D paged kernel loads one physical
   block index per tile and derives the 16 in-page offsets arithmetically,
   instead of repeating division, remainder and page-table loads in every
   lane. K/V addresses and the FP32 softmax reduction order are unchanged.
10. **Length-specialized decode softmax.** Single-request decode uses 128
   independent segments through 24K and 64 beyond that point, with a measured
   two-warp, one-stage gfx936 launch. A graph-stable 128-segment allocation is
   shared by both paths; the long path reads and writes only its first half.
   This avoids the generic kernel's occupancy cliff while keeping all segment
   statistics and the final reduction in FP32. It is an explicitly
   higher-throughput candidate because segment partitioning changes
   floating-point reduction grouping.
11. **Processor-free greedy sampler fast path.** On gfx936, contiguous BF16
   logits for a request that is entirely greedy, asks for no logprobs and has
   no penalties, masks, bad words or non-argmax-invariant processors go
   directly to `argmax`. This avoids a redundant full-vocabulary FP32 copy.
   Every feature-bearing sampling path retains the original FP32 processing.
12. **Exact GDN gate preparation plus local cumsum.** For one 4096-token
   Qwen3.5 BF16 sequence, the post-convolution preparation kernel gives each
   gate program one complete 64-token recurrent chunk and writes cumulative
   FP32 decay directly. This removes the separate local-cumsum launch and its
   HBM round trip while preserving Q/K/V, beta and cumulative gate bit for
   bit. A shared predicate covers gfx936, model geometry, dtype, device,
   layout and the single-sequence boundary; multi-request and varlen batches
   retain the original cumsum.
13. **Exact GDN triangular solve plus W/U recomputation.** The guarded direct
   4096-token inference route solves each 64x64 lower-triangular block and
   immediately emits W/U from the ten 16x16 inverse blocks. It explicitly
   recreates the original BF16 RTNE inverse boundary, so W/U, attention output
   and recurrent state are bitwise identical while removing one launch and
   the solved inverse tensor's store/reload. Public, autograd, varlen and
   non-direct-output paths remain unchanged.
14. **Per-shape BF16 prefill BLAS selection.** Only the exact
   `[4096,5120] @ [34816,5120]` gate/up projection selects hipBLASLt, which is
   faster and bitwise identical on gfx936. The caller's prior backend is saved
   and restored, and batch-invariant execution disables this specialization;
   all other projections retain their measured faster backend.
15. **Exact causal-convolution plus GDN preparation.** The single-sequence
   4096-token gfx936 route computes the width-4 SiLU convolution directly into
   raw BF16 Q/K, V, beta and cumulative gate outputs, updates the original
   convolution cache, then normalizes Q/K in a small second kernel. It retains
   the original BF16 convolution store/load boundary and is bitwise identical
   with and without an initial state. Platform, sequence count, geometry,
   activation, bias, cache layout, dtype and stride guards preserve the mature
   convolution path for every other case.

## Validation and screening evidence

The last public evaluation of the clean BF16 parent
`pra26-flash784-cont-pagefast-gdn-memory` was AC with no SLA or accuracy
deduction: 19.46 / 17.99 / 14.85 tok/s for 4-8K / 8-16K / 16-32K. These are
public parent results, not a claim for this branch.

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

For the new dense gate/up fusion, the exact M=1 `[1,5120] @ [34816,5120]`
remote microbenchmark measured approximately 0.3211 ms for AITER GEMV plus
the separate activation and 0.3129--0.3140 ms for the fused operation. M=1,
the M=2 fallback and a full-graph custom-op call all matched their respective
BF16 references bit for bit on the screening tensors.

A same-code, same-node serving A/B changed only
`VLLM_ROCM_QWEN35_FUSED_GATEUP_SILU`. With two fixed prompts per bucket and
two warmups, enabling the fusion reduced P99 TPOT from 47.20 / 47.55 / 48.07
ms to 46.67 / 47.06 / 47.66 ms. All six generated texts and output lengths
(150 / 104 / 288 tokens by bucket) were identical. The small prompt count is
a compatibility and direction screen, not a substitute for official scoring.

An end-to-end OpenCompass smoke run then evaluated two rows from each official
task family. It completed without service errors and reported HotpotQA 100.00,
GovReport 39.79, retrieval-multi-point 100.00 and keyword-aggregation 100.00.
This verifies all four accuracy routes and output parsing; the official full
hidden-set evaluation remains authoritative.

The new exact 4096-token gate+cumsum path reduced its isolated preparation
from about 0.3859 to 0.3683 ms (1.048x). The exact fused triangular
solve/recompute reduced that pair from 0.5954 to 0.5272 ms and the complete
GDN recurrent core from 1.8486 to 1.7822 ms (3.59%). The shape-local
hipBLASLt gate/up call, including backend selection and restoration, reduced
the isolated projection from about 4.4359 to 4.4149 ms. All three comparisons
were run on the target gfx936 and passed bitwise output/state checks.

The exact convolution/preparation pair reduced 0.5737 to 0.5484 ms per GDN
layer (4.60%) and matched Q/K/V/g/beta plus the complete convolution cache bit
for bit for both initial-state modes.

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
- Fusing KKT construction into the new solve/recompute kernel: splitting one
  efficient 64x64 dot into ten 16x16 dots made the complete recurrent core
  about 7.2% slower and introduced sparse rounding changes, so the exact
  two-stage KKT plus fused-solve design is retained.
- A custom 4096-token gate/up GEMM plus SwiGLU kernel: despite avoiding the
  activation intermediate, it was roughly three times slower than the tuned
  BLAS GEMM plus activation and was rejected.
- FlashAttention 3, direct AITER MHA and the installed Triton varlen-attention
  alternative: unsupported or incompatible in the supplied software stack.
  The measured dense ROCm FlashAttention 2 continuation path remains fastest.
- Persistent gate/up INT8, persistent lm-head quantization and load-time
  reusable low-precision weight copies: rejected as outside the permitted
  optimization scope. This branch contains none of these paths.

## Source locations

- `vllm/model_executor/layers/utils.py`: exact GEMV dispatch.
- `vllm/model_executor/layers/qwen35_gateup_silu.py`: guarded BF16 gated GEMV
  plus SwiGLU fusion.
- `vllm/model_executor/layers/fla/ops/`: guarded GDN fusion and recurrent ops.
- `vllm/model_executor/layers/fused_qk_norm_rope.py`: Q/K norm, gate and RoPE.
- `vllm/model_executor/models/qwen3_5.py` and `qwen3_next.py`: model routing.
- `vllm/v1/attention/backends/triton_attn.py`: phase-specific attention route
  and builder-owned dense scratch.
- `vllm/v1/attention/ops/triton_qwen35_kv_gather.py`: exact paged K/V gather.
- `vllm/v1/attention/ops/triton_unified_attention.py`: guarded fallback tile.
- `vllm/v1/sample/sampler.py`: processor-free BF16 greedy fast path.
- `vllm/platforms/rocm.py`: isolated gfx936 predicate.
