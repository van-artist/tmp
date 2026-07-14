# SPDX-License-Identifier: Apache-2.0
"""Standalone hipBLAS-vs-fused Qwen3.5 prefill Gate-Up GEMM test."""

import argparse

import torch
from triton.testing import do_bench

from vllm.model_executor.layers import utils as _linear_utils  # noqa: F401
from vllm.model_executor.layers import qwen35_gateup_swiglu_gemm  # noqa: F401

K = 5120
N = 17408


def make_inputs(tokens: int, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    x = torch.empty((tokens, K), device=device, dtype=torch.bfloat16)
    x.normal_().mul_(0.02)
    weight = torch.empty((2 * N, K), device=device, dtype=torch.bfloat16)
    weight.normal_().mul_(0.01)
    return x, weight


def hipblas_swiglu(x: torch.Tensor, weight: torch.Tensor):
    # For M=4096/8192 this dispatch falls through to F.linear/hipBLAS.
    gate_up = torch.ops.vllm.rocm_unquantized_gemm(x, weight, None)
    output = torch.empty((x.shape[0], N), device=x.device, dtype=x.dtype)
    torch.ops._C.silu_and_mul(output, gate_up)
    return output


def fused_gemm(x: torch.Tensor, weight: torch.Tensor):
    return torch.ops.vllm.qwen35_gateup_swiglu_gemm(x, weight)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(4096, 8192), default=4096)
    parser.add_argument("--seed", type=int, default=13)
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
        f"allocating Gate-Up GEMM M={args.tokens}, N={2 * N}, K={K}..."
    )
    x, weight = make_inputs(args.tokens, args.seed)
    print("JIT/autotuning and checking fused epilogue...")
    reference = hipblas_swiglu(x, weight)
    actual = fused_gemm(x, weight)
    torch.cuda.synchronize()
    diff = (actual.float() - reference.float()).abs()
    print(
        f"max_abs={diff.max().item():.8f}, "
        f"mean_abs={diff.mean().item():.8f}"
    )
    torch.testing.assert_close(actual, reference, atol=0.125, rtol=0.02)
    print("correctness=PASS")

    if args.correctness_only:
        return

    baseline_ms = do_bench(lambda: hipblas_swiglu(x, weight))
    fused_ms = do_bench(lambda: fused_gemm(x, weight))
    print(f"hipblas_gemm_plus_swiglu={baseline_ms:.3f} ms")
    print(f"fused_gateup_swiglu_gemm={fused_ms:.3f} ms")
    print(f"speedup={baseline_ms / fused_ms:.3f}x")
    print(
        f"estimated_64_layer_saving="
        f"{(baseline_ms - fused_ms) * 64:.3f} ms/chunk"
    )


if __name__ == "__main__":
    main()
