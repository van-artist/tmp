# SPDX-License-Identifier: Apache-2.0
"""Standalone Qwen3.5 LM-head GEMV + greedy argmax comparison."""

import argparse

import torch
from triton.testing import do_bench

from vllm.model_executor.layers import utils as _linear_utils  # noqa: F401
from vllm.model_executor.layers import qwen35_lm_head_argmax  # noqa: F401

HIDDEN_SIZE = 5120
DEFAULT_VOCAB_SIZE = 248320


def make_inputs(seed: int, vocab_size: int):
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    x = torch.randn(
        (1, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
    ) * 0.1
    weight = torch.empty(
        (vocab_size, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
    )
    weight.normal_().mul_(0.01)
    return x, weight


def hipblas_logits(x, weight):
    # This is the same unquantized dispatch used by ParallelLMHead.  The LM
    # shape is not one of the three AITER-specialized shapes, so gfx936 falls
    # back to F.linear/hipBLAS.
    return torch.ops.vllm.rocm_unquantized_gemm(x, weight, None)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is not available")
    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", "unknown"))
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"arch={arch}")
    if arch.split(":", 1)[0] != "gfx936":
        raise RuntimeError(f"expected gfx936, got {arch}")

    print(
        f"allocating LM head [{args.vocab_size}, {HIDDEN_SIZE}] BF16 "
        f"({args.vocab_size * HIDDEN_SIZE * 2 / 2**30:.2f} GiB)..."
    )
    x, weight = make_inputs(args.seed, args.vocab_size)

    print("JIT compiling and checking greedy token...")
    logits = hipblas_logits(x, weight)
    baseline_fp32 = logits.float().argmax(dim=-1).to(torch.int32)
    baseline_bf16 = logits.argmax(dim=-1).to(torch.int32)
    fused = torch.ops.vllm.qwen35_lm_head_argmax(x, weight)
    torch.cuda.synchronize()
    print(f"hipblas_fp32_argmax={baseline_fp32.item()}")
    print(f"hipblas_bf16_argmax={baseline_bf16.item()}")
    print(f"fused_streaming_argmax={fused.item()}")
    if not torch.equal(baseline_fp32, baseline_bf16):
        raise AssertionError("BF16 to FP32 copy unexpectedly changed argmax")
    if not torch.equal(baseline_fp32, fused):
        raise AssertionError("streaming fused kernel selected a different token")
    print("correctness=PASS")

    if args.correctness_only:
        return

    def run_hipblas_fp32():
        return hipblas_logits(x, weight).float().argmax(dim=-1)

    def run_hipblas_bf16():
        return hipblas_logits(x, weight).argmax(dim=-1)

    def run_fused():
        return torch.ops.vllm.qwen35_lm_head_argmax(x, weight)

    print("benchmarking warmed paths...")
    fp32_ms = do_bench(run_hipblas_fp32)
    bf16_ms = do_bench(run_hipblas_bf16)
    fused_ms = do_bench(run_fused)
    print(f"hipblas_logits_fp32_argmax={fp32_ms * 1000:.2f} us")
    print(
        f"hipblas_logits_bf16_argmax={bf16_ms * 1000:.2f} us, "
        f"speedup={fp32_ms / bf16_ms:.3f}x"
    )
    print(
        f"fused_streaming_gemv_argmax={fused_ms * 1000:.2f} us, "
        f"speedup={fp32_ms / fused_ms:.3f}x"
    )


if __name__ == "__main__":
    main()
