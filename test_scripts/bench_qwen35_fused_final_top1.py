#!/usr/bin/env python3
"""Benchmark Qwen3.5 final RMSNorm, lm_head GEMV, and greedy top-1."""

from __future__ import annotations

import argparse
import csv
import importlib
import time
from collections.abc import Callable
from pathlib import Path

import torch

from vllm import _custom_ops as ops


DEFAULT_M = 248320
DEFAULT_K = 5120
PROBE_M = 4096
EPSILON = 1e-6


def gemma_norm(
    hidden: torch.Tensor, norm_weight: torch.Tensor
) -> torch.Tensor:
    value = hidden.float()
    inverse_rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + EPSILON)
    return (value * inverse_rms * (1.0 + norm_weight.float())).to(hidden.dtype)


def time_ms(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def token_id(value: torch.Tensor) -> int:
    if value.numel() != 1:
        raise RuntimeError(f"expected one token id, got {value.shape}")
    return int(value.item())


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = (actual.float() - expected.float()).norm()
    denominator = expected.float().norm().clamp_min(1e-12)
    return float((delta / denominator).item())


def output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def workspaces(
    m: int, k: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((1, k), device=device, dtype=torch.bfloat16),
        torch.empty(m // 2, device=device, dtype=torch.float32),
        torch.empty(m // 2, device=device, dtype=torch.int32),
        torch.full((1,), -1, device=device, dtype=torch.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--prefill-tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--validation-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reduction-blocks", type=int, default=80)
    parser.add_argument(
        "--output", type=Path, default=Path("qwen35_fused_final_top1.csv")
    )
    args = parser.parse_args()

    if args.m <= 0 or args.m % 2 or args.k <= 0 or args.k % 8:
        raise SystemExit("M must be even and K must be divisible by eight")
    if (
        args.prefill_tokens <= 0
        or args.warmup < 0
        or args.iters <= 0
        or args.validation_iters <= 0
    ):
        raise SystemExit(
            "prefill tokens and benchmark/validation iterations must be positive"
        )
    if args.reduction_blocks <= 0 or args.reduction_blocks > PROBE_M // 2:
        raise SystemExit("reduction blocks must be in [1, PROBE_M/2]")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA/HIP device is required")

    importlib.import_module("vllm._rocm_C")
    required_ops = (
        "LLMM1",
        "LLMM1Top1",
        "LLMM1RMSNormTop1",
    )
    missing = [name for name in required_ops if not hasattr(torch.ops._rocm_C, name)]
    if missing:
        raise SystemExit(f"Missing ROCm operators: {', '.join(missing)}")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    properties = torch.cuda.get_device_properties(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    compiled_norm = torch.compile(gemma_norm, fullgraph=True)

    probe_weight = torch.zeros(PROBE_M, args.k, device=device, dtype=dtype)
    probe_hidden = torch.ones(1, args.k, device=device, dtype=dtype)
    probe_norm_weight = torch.zeros(args.k, device=device, dtype=dtype)
    probe_workspace = workspaces(PROBE_M, args.k, device)

    def probe_candidate() -> torch.Tensor:
        normalized, values, indices, result = probe_workspace
        ops.LLMM1RMSNormTop1(
            probe_weight,
            probe_hidden,
            probe_norm_weight,
            normalized,
            values,
            indices,
            result,
            EPSILON,
            args.reduction_blocks,
        )
        return result

    probe_weight[0].fill_(1)
    probe_weight[1].fill_(1)
    probe_weight[1, 0] = 2
    probe_normalized = compiled_norm(probe_hidden, probe_norm_weight)
    bf16_probe_token = token_id(ops.LLMM1(probe_weight, probe_normalized, 2).argmax())
    fused_probe_token = token_id(probe_candidate())
    fp32_probe_token = token_id(
        torch.mv(probe_weight.float(), probe_normalized[0].float()).argmax()
    )
    rounding_probe_pass = (
        bf16_probe_token == 0 and fused_probe_token == 0 and fp32_probe_token == 1
    )

    probe_weight.zero_()
    probe_hidden.zero_()
    zero_baseline_token = token_id(
        ops.LLMM1(
            probe_weight, compiled_norm(probe_hidden, probe_norm_weight), 2
        ).argmax()
    )
    zero_candidate_token = token_id(probe_candidate())
    zero_tie_pass = zero_baseline_token == zero_candidate_token == 0
    probe_hidden.fill_(float("nan"))
    nan_baseline_token = token_id(
        ops.LLMM1(
            probe_weight, compiled_norm(probe_hidden, probe_norm_weight), 2
        ).argmax()
    )
    nan_candidate_token = token_id(probe_candidate())
    nan_match_pass = nan_baseline_token == nan_candidate_token

    probe_weight.normal_()
    probe_norm_weight.normal_(std=0.1)
    baseline_mismatches = 0
    worst_norm_max_abs = 0.0
    worst_norm_relative_l2 = 0.0
    for _ in range(args.validation_iters):
        probe_hidden.normal_()
        expected_normalized = compiled_norm(probe_hidden, probe_norm_weight)
        expected_token = token_id(
            ops.LLMM1(probe_weight, expected_normalized, 2).argmax()
        )
        actual_token = token_id(probe_candidate())
        actual_normalized = probe_workspace[0]
        worst_norm_max_abs = max(
            worst_norm_max_abs,
            float(
                (actual_normalized.float() - expected_normalized.float())
                .abs()
                .max()
                .item()
            ),
        )
        worst_norm_relative_l2 = max(
            worst_norm_relative_l2,
            relative_l2(actual_normalized, expected_normalized),
        )
        baseline_mismatches += actual_token != expected_token

    del probe_weight, probe_hidden, probe_norm_weight, probe_workspace
    torch.cuda.empty_cache()

    weight = torch.randn(args.m, args.k, device=device, dtype=dtype)
    hidden = torch.randn(1, args.k, device=device, dtype=dtype)
    norm_weight = torch.randn(args.k, device=device, dtype=dtype) * 0.1
    normalized, local_values, local_indices, result = workspaces(
        args.m, args.k, device
    )

    def standard_tail() -> torch.Tensor:
        normalized_hidden = compiled_norm(hidden, norm_weight)
        return ops.LLMM1(weight, normalized_hidden, 2).float().argmax()

    def top1_tail() -> torch.Tensor:
        normalized_hidden = compiled_norm(hidden, norm_weight)
        ops.LLMM1Top1(
            weight,
            normalized_hidden,
            local_values,
            local_indices,
            result,
            args.reduction_blocks,
        )
        return result

    def fused_tail() -> torch.Tensor:
        ops.LLMM1RMSNormTop1(
            weight,
            hidden,
            norm_weight,
            normalized,
            local_values,
            local_indices,
            result,
            EPSILON,
            args.reduction_blocks,
        )
        return result

    top1_baseline_mismatches = 0
    fused_baseline_mismatches = 0
    for _ in range(args.validation_iters):
        hidden.normal_()
        baseline_token = token_id(standard_tail())
        top1_baseline_mismatches += token_id(top1_tail()) != baseline_token
        fused_baseline_mismatches += token_id(fused_tail()) != baseline_token

    standard_token = token_id(standard_tail())
    top1_token = token_id(top1_tail())
    fused_token = token_id(fused_tail())
    standard_before_ms = time_ms(standard_tail, args.warmup, args.iters)
    top1_ms = time_ms(top1_tail, args.warmup, args.iters)
    fused_ms = time_ms(fused_tail, args.warmup, args.iters)
    standard_after_ms = time_ms(standard_tail, args.warmup, args.iters)
    standard_mean_ms = (standard_before_ms + standard_after_ms) / 2

    prefill_hidden = torch.randn(
        args.prefill_tokens, args.k, device=device, dtype=dtype
    )

    def full_prefill_tail() -> torch.Tensor:
        normalized_rows = compiled_norm(prefill_hidden, norm_weight)
        ops.LLMM1Top1(
            weight,
            normalized_rows[-1:],
            local_values,
            local_indices,
            result,
            args.reduction_blocks,
        )
        return result

    def deferred_prefill_tail() -> torch.Tensor:
        ops.LLMM1RMSNormTop1(
            weight,
            prefill_hidden[-1:],
            norm_weight,
            normalized,
            local_values,
            local_indices,
            result,
            EPSILON,
            args.reduction_blocks,
        )
        return result

    full_prefill_token = token_id(full_prefill_tail())
    deferred_prefill_token = token_id(deferred_prefill_tail())
    full_prefill_before_ms = time_ms(full_prefill_tail, args.warmup, args.iters)
    deferred_prefill_ms = time_ms(
        deferred_prefill_tail, args.warmup, args.iters
    )
    full_prefill_after_ms = time_ms(full_prefill_tail, args.warmup, args.iters)
    full_prefill_mean_ms = (full_prefill_before_ms + full_prefill_after_ms) / 2

    component_speedup = top1_ms / fused_ms
    standard_speedup = standard_mean_ms / fused_ms
    prefill_speedup = full_prefill_mean_ms / deferred_prefill_ms
    validation_pass = (
        rounding_probe_pass
        and zero_tie_pass
        and nan_match_pass
        and baseline_mismatches == 0
        and top1_baseline_mismatches == 0
        and fused_baseline_mismatches == 0
        and standard_token == top1_token == fused_token
        and full_prefill_token == deferred_prefill_token
        and worst_norm_relative_l2 <= 1e-3
    )
    performance_pass = standard_speedup >= 1.005 and prefill_speedup >= 1.0
    status = "COMPONENT_PASS" if validation_pass and performance_pass else "STOP"

    print(
        f"device={properties.name}, compute_units={properties.multi_processor_count}, "
        f"shape=({args.m}, {args.k}), dtype={dtype}"
    )
    print(
        "rounding probe: "
        f"BF16={bf16_probe_token}, fused={fused_probe_token}, "
        f"FP32={fp32_probe_token}, pass={rounding_probe_pass}"
    )
    print(
        "validation: "
        f"zero_tie={zero_baseline_token}/{zero_candidate_token}, "
        f"nan_match={nan_baseline_token}/{nan_candidate_token}, "
        f"baseline_mismatches={baseline_mismatches}/{args.validation_iters}, "
        f"full_vocab_top1_mismatches={top1_baseline_mismatches}/"
        f"{args.validation_iters}, "
        f"full_vocab_fused_mismatches={fused_baseline_mismatches}/"
        f"{args.validation_iters}, "
        f"norm_max_abs={worst_norm_max_abs:.8g}, "
        f"norm_rel_l2={worst_norm_relative_l2:.8g}"
    )
    print(
        "decode tail: standard/top1/fused/standard="
        f"{standard_before_ms:.4f}/{top1_ms:.4f}/{fused_ms:.4f}/"
        f"{standard_after_ms:.4f} ms; fused_vs_top1={component_speedup:.4f}x; "
        f"fused_vs_standard={standard_speedup:.4f}x; "
        f"tokens={standard_token}/{top1_token}/{fused_token}"
    )
    print(
        f"prefill rows={args.prefill_tokens}: full/deferred/full="
        f"{full_prefill_before_ms:.4f}/{deferred_prefill_ms:.4f}/"
        f"{full_prefill_after_ms:.4f} ms; speedup={prefill_speedup:.4f}x; "
        f"tokens={full_prefill_token}/{deferred_prefill_token}; gate={status}"
    )

    path = output_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "m": args.m,
        "k": args.k,
        "prefill_tokens": args.prefill_tokens,
        "validation_iters": args.validation_iters,
        "reduction_blocks": args.reduction_blocks,
        "standard_before_ms": f"{standard_before_ms:.6f}",
        "top1_ms": f"{top1_ms:.6f}",
        "fused_ms": f"{fused_ms:.6f}",
        "standard_after_ms": f"{standard_after_ms:.6f}",
        "fused_vs_top1_speedup": f"{component_speedup:.6f}",
        "fused_vs_standard_speedup": f"{standard_speedup:.6f}",
        "full_prefill_before_ms": f"{full_prefill_before_ms:.6f}",
        "deferred_prefill_ms": f"{deferred_prefill_ms:.6f}",
        "full_prefill_after_ms": f"{full_prefill_after_ms:.6f}",
        "prefill_speedup": f"{prefill_speedup:.6f}",
        "standard_token": standard_token,
        "top1_token": top1_token,
        "fused_token": fused_token,
        "full_prefill_token": full_prefill_token,
        "deferred_prefill_token": deferred_prefill_token,
        "rounding_probe_pass": rounding_probe_pass,
        "zero_tie_pass": zero_tie_pass,
        "nan_match_pass": nan_match_pass,
        "nan_baseline_token": nan_baseline_token,
        "nan_candidate_token": nan_candidate_token,
        "baseline_mismatches": baseline_mismatches,
        "top1_baseline_mismatches": top1_baseline_mismatches,
        "fused_baseline_mismatches": fused_baseline_mismatches,
        "worst_norm_max_abs": f"{worst_norm_max_abs:.10g}",
        "worst_norm_relative_l2": f"{worst_norm_relative_l2:.10g}",
        "validation_pass": validation_pass,
        "performance_pass": performance_pass,
        "status": status,
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
