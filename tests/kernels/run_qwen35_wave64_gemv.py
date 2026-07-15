#!/usr/bin/env python3
"""Standalone gfx936 Qwen3.5 large-K GEMV correctness and latency check."""

import math
import statistics

import torch

from vllm import _custom_ops as ops

SHAPES = (
    ("mlp_down", 5120, 17408),
    ("gdn_qkvzba", 16480, 5120),
)


def measure_us(fn, warmup: int = 20, repeats: int = 7, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    return statistics.median(samples)


@torch.inference_mode()
def run_case(name: str, rows: int, width: int) -> None:
    scale = 1.0 / math.sqrt(width)
    x = torch.randn((1, width), device="cuda", dtype=torch.bfloat16) * scale
    weight = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)

    reference = torch.nn.functional.linear(x, weight)
    candidate = ops.qwen35_wave64_gemv(weight, x)
    diff = (candidate.float() - reference.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    torch.testing.assert_close(candidate, reference, atol=1.5625e-2, rtol=2.0e-2)

    baseline_us = measure_us(lambda: torch.nn.functional.linear(x, weight))
    wave64_us = measure_us(lambda: ops.qwen35_wave64_gemv(weight, x))
    print(
        f"{name}: shape=(1,{width})x({rows},{width}) "
        f"max_abs={max_abs:.8f} mean_abs={mean_abs:.8f}"
    )
    print(
        f"  hipblas={baseline_us:.2f} us wave64={wave64_us:.2f} us "
        f"speedup={baseline_us / wave64_us:.3f}x"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A ROCm device is required")
    if not hasattr(torch.ops._rocm_C, "qwen35_wave64_gemv"):
        raise RuntimeError("Rebuild/install the wheel: custom op is missing")
    props = torch.cuda.get_device_properties(0)
    print(f"device={props.name}")
    for case in SHAPES:
        run_case(*case)
    print("correctness=PASS")


if __name__ == "__main__":
    main()
