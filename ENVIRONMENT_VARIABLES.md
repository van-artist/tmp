# Environment variables

The high-score defaults are compiled into this branch, so correctness and the
core speedups do not depend on external variable injection. The following
values document the tested launch and provide explicit rollback controls.

| Variable | Tested/default value | Purpose |
|---|---:|---|
| `VLLM_ROCM_USE_GFX936_AITER_GEMV` | `1` | Enable the three exact M=1 BF16 GEMVs; `0` restores PyTorch linear. |
| `VLLM_ROCM_USE_GFX936_FUSED_SWIGLU` | `1` | Fuse exact Qwen3.5 decode gate/up projection with SwiGLU; `0` restores the separate projection and activation path. |
| `VLLM_ROCM_USE_AITER_TRITON_GEMM` | `1` | Permit AITER Triton GEMM. |
| `VLLM_ROCM_USE_AITER` | `0` | Keep unrelated global AITER routes disabled. |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `0` | Do not use the rejected gfx936 skinny route. |
| `VLLM_ROCM_PREFIX_PREFILL_TILE_M` | `64` | Query tile for the exact Qwen3.5 geometry. |
| `VLLM_ROCM_PREFIX_PREFILL_TILE_N` | `32` | Current-query key tile. |
| `VLLM_ROCM_PREFIX_PREFILL_NUM_WARPS` | `4` | Triton launch warps for the exact geometry. |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `0` | Retain the faster hipBLAS path for measured fallback shapes. |

The three `VLLM_ROCM_PREFIX_*` values are validated at import. Overrides only
apply to ROCm BF16 with page 784 and head dimension 256; other geometries keep
upstream defaults.
