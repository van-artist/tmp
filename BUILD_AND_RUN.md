# Build and run

## Offline wheel build

Use the fixed competition Python 3.10 / PyTorch 2.10 / DTK 26.04 image:

```bash
python setup.py bdist_wheel
python -m pip install --no-deps --force-reinstall dist/vllm-*.whl
```

No model, test data, prebuilt wheel, binary extension or generated cache is
stored in this branch.

## Service

```bash
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_TRITON_GEMM=1
export VLLM_ROCM_USE_GFX936_AITER_GEMV=1
export VLLM_ROCM_USE_GFX936_FUSED_SWIGLU=1
export VLLM_ROCM_USE_SKINNY_GEMM=0
export VLLM_ROCM_PREFIX_PREFILL_TILE_M=64
export VLLM_ROCM_PREFIX_PREFILL_TILE_N=32
export VLLM_ROCM_PREFIX_PREFILL_NUM_WARPS=4
export TORCH_BLAS_PREFER_HIPBLASLT=0

vllm serve /path/to/Qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --port 8001 --trust-remote-code --dtype bfloat16 \
  --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 128 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.95 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

The scheduler and locked competition arguments must not be changed.

## Validation

Run the supplied official scripts and require zero failures/empty errors:

```bash
./run_throughput.sh 4-8K 50
./run_throughput.sh 8-16K 50
./run_throughput.sh 16-32K 50
./run_accuracy.sh all
```

Check health, completed count, output token sum, P99 TTFT/TPOT and accuracy
before choosing the high-score branch. Set
`VLLM_ROCM_USE_GFX936_AITER_GEMV=0` for a numerical rollback comparison.
