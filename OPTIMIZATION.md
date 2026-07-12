# PRA26 optimization report

## Scope and invariants

Target: Qwen3.5-27B BF16, one gfx936 DCU, tensor parallel size 1 and
concurrency 1. The official scheduler, `max_num_batched_tokens=4096`,
`max_num_seqs=128`, model weights, layer count, dense attention semantics and
sampling workload are unchanged. No speculative decoding, pruning, sparse or
approximate attention, skipped work, retained low-precision weight copy, or
checkpoint conversion is used.

## Retained optimizations

1. **Exact-shape gfx936 BF16 GEMV dispatch.** Three M=1 projections use
   AITER's `gemm_a16w16` Triton kernel with measured static configurations:
   `(34816,5120)` gate/up, `(16384,5120)` GDN QKVZ, and `(14336,5120)` full
   attention QKV. Dispatch requires gfx936, BF16, M=1, no bias, equal
   dtype/device and contiguous tensors. Missing AITER APIs and every other
   shape fall back to `torch.nn.functional.linear`. Weights stay BF16 in their
   original layout.
2. **Phase-specific dense attention.** TRITON_ATTN remains the decode backend.
   Only single-request Qwen3.5 BF16 decoder prefill (`max_seqlen_q > 1`) on
   gfx936 calls vLLM's dense ROCm prefix-prefill kernel through zero-copy K/V
   cache views. The route additionally requires Hq/Hkv/D=24/4/256, page 784,
   no softcap/MM prefix/quantized output, and BF16 K/V. Mixed batches,
   cross-attention and every other geometry fall back to upstream unified
   attention. K/V storage and page geometry are unchanged.
3. **gfx936 Qwen3.5 prefix tile.** For BF16, head dimension 256 and physical
   page 784 only, query tile M changes from 32 to 64. Key tile, logical cache
   traversal tile and physical page remain 32/32/784. This halves query-side
   programs without changing any attention element or causal mask. Four warps
   is retained; 8 warps and M=128 were measured and rejected.

## Measured screening results

All numbers below use the official script, two warmups, the same first five
prompts and the same gfx936 software stack. They are screening results rather
than claimed hidden-set scores.

| Candidate | 4-8K tok/s | 8-16K tok/s | 16-32K tok/s | mean TPOT | mean long TTFT |
|---|---:|---:|---:|---:|---:|
| Official BF16 baseline | 8.83 | 7.14 | 3.61 | 68.7-71.4 ms | 26.63 s |
| Hybrid prefill, 32x32 | 9.91 | - | 4.21 | unchanged | 22.47 s |
| AITER3 + hybrid 32x32 | 12.17 | 9.06 | 4.57 | 49.75-52.48 ms | 22.45 s |
| AITER3 + hybrid 64x32 | 13.01 | - | 6.12 | 49.50-52.57 ms | 14.91 s |

Microbenchmarks for the retained M=1 routes:

| Projection | torch BF16 | tuned AITER | speedup |
|---|---:|---:|---:|
| gate/up 34816x5120 | 0.501 ms | 0.311 ms | 1.61x |
| GDN QKVZ 16384x5120 | 0.241 ms | 0.151 ms | 1.60x |
| full QKV 14336x5120 | 0.237 ms | 0.134 ms | 1.77x |

Each microbenchmark was finite and deterministic; the close fraction against
PyTorch was 100% at `rtol=0.02, atol=0.01`. End-to-end screening completed all
requests with empty error fields. The 64x32 tile preserved the 32x32 generated
texts on the compared long prompts. Its clean short-bucket run generated the
same 386 tokens as baseline across the first five prompts, with 5/5 successful
requests and zero failures. Full public accuracy remains the final release gate
because GEMM reduction order is not bit-identical.

## Rejected experiments

- Batch-1 static compilation (`compile_sizes=[1]`): engine initialization
  failed after compiling the singleton and dynamic ranges. The proven default
  FULL_AND_PIECEWISE graph configuration is retained.
- gfx936 `wvSplitK`: compiled only after replacing an unsupported dot
  instruction, but failed numerical validation (91.4% outside tolerance).
- Full ROCM_ATTN: prefill improved but decode TPOT regressed to 117-173 ms.
- Prefix tiles 64x64 and 128x32: slower and/or changed generations.
- 64x32 with 8 warps: first long TTFT 19.29 s versus 13.87 s with 4 warps.
- Global hipBLASLt preference: large GDN/full-QKV GEMMs regressed 39.6%/24.7%.
- AITER for down/out/BA/lm_head: slower; lm_head also failed the numerical gate.

## Source locations

- `vllm/model_executor/layers/utils.py`: exact GEMV configs and dispatch.
- `vllm/platforms/rocm.py`: isolated gfx936 predicate.
- `vllm/v1/attention/backends/triton_attn.py`: prefill/decode dense dispatch.
- `vllm/v1/attention/ops/prefix_prefill.py`: exact geometry tile selection.
- `vllm/envs.py`: documented, validated opt-out/tuning variables.
