# Environment variables

The high-score defaults are compiled into this branch, so correctness and the
core speedups do not depend on external variable injection. The following
values document the tested launch and provide explicit rollback controls.

| Variable | Tested/default value | Purpose |
|---|---:|---|
| `VLLM_ROCM_QWEN35_FUSED_GATEUP_SILU` | `1` | Fuse the exact BF16 M=1 gate/up GEMV and SwiGLU; `0` restores separate BF16 operations. |
| `VLLM_ROCM_USE_GFX936_AITER_GEMV` | `1` | Enable the exact-shape BF16 GEMV allowlist; `0` restores PyTorch linear. |
| `VLLM_ROCM_USE_AITER_TRITON_GEMM` | `1` | Permit AITER Triton GEMM. |
| `VLLM_ROCM_USE_AITER` | `0` | Keep unrelated global AITER routes disabled. |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `0` | Do not use the rejected gfx936 skinny route. |
| `VLLM_ROCM_PREFIX_PREFILL_TILE_M` | `64` | Query tile for the exact Qwen3.5 geometry. |
| `VLLM_ROCM_PREFIX_PREFILL_TILE_N` | `32` | Current-query key tile. |
| `VLLM_ROCM_PREFIX_PREFILL_NUM_WARPS` | `4` | Triton launch warps for the exact geometry. |
| `VLLM_ROCM_QWEN35_UNIFIED_BLOCK_M` | `64` | Fallback paged-attention query-head tile. |
| `VLLM_ROCM_QWEN35_UNIFIED_TILE_SIZE` | `32` | Fallback paged-attention key/value tile. |
| `VLLM_ROCM_QWEN35_UNIFIED_NUM_WARPS` | `4` | Fallback paged-attention launch warps. |
| `VLLM_ROCM_QWEN35_UNIFIED_NUM_STAGES` | `1` | Fallback paged-attention pipeline stages. |
| `VLLM_ROCM_QWEN35_UNIFIED_WAVES_PER_EU` | `1` | Fallback paged-attention wave occupancy hint. |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `0` | Retain the faster hipBLAS path for measured fallback shapes. |

The three `VLLM_ROCM_PREFIX_*` values are validated at import. The five
`VLLM_ROCM_QWEN35_UNIFIED_*` values are validated against finite allowlists.
Overrides only apply to ROCm BF16 Qwen3.5 with page 784 and head dimension 256;
other geometries keep upstream defaults.
