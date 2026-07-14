#!/usr/bin/env python3
"""Benchmark Qwen3.5 lm_head top-1 in the LLMM1 FP32 accumulator domain."""

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


def output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def workspaces(m: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty(m // 2, device=device, dtype=torch.float32),
        torch.empty(m // 2, device=device, dtype=torch.int32),
        torch.full((1,), -1, device=device, dtype=torch.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reduction-blocks",
        type=int,
        nargs="+",
        default=None,
        help="first-stage reduction block counts; default is 1/2/4 times CU",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("qwen35_lm_head_fp32_top1.csv")
    )
    args = parser.parse_args()

    if args.m <= 0 or args.m % 2 or args.k <= 0 or args.k % 8:
        raise SystemExit("M must be even and K must be divisible by eight")
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be non-negative and --iters positive")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA/HIP device is required")

    importlib.import_module("vllm._rocm_C")
    missing = [
        name
        for name in ("LLMM1", "LLMM1FP32Top1")
        if not hasattr(torch.ops._rocm_C, name)
    ]
    if missing:
        raise SystemExit(f"Missing ROCm operators: {', '.join(missing)}")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    properties = torch.cuda.get_device_properties(device)
    reduction_blocks = sorted(
        set(
            args.reduction_blocks
            or [
                properties.multi_processor_count * multiplier
                for multiplier in (1, 2, 4)
            ]
        )
    )
    if (
        not reduction_blocks
        or reduction_blocks[0] <= 0
        or reduction_blocks[-1] > min(args.m // 2, PROBE_M // 2)
    ):
        raise SystemExit(
            "--reduction-blocks must be in [1, min(M/2, PROBE_M/2)]"
        )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    probe_weight = torch.zeros(PROBE_M, args.k, device=device, dtype=dtype)
    probe_hidden = torch.ones(1, args.k, device=device, dtype=dtype)
    probe_values, probe_indices, probe_result = workspaces(PROBE_M, device)

    def probe_candidate(blocks: int) -> torch.Tensor:
        ops.LLMM1FP32Top1(
            probe_weight,
            probe_hidden,
            probe_values,
            probe_indices,
            probe_result,
            blocks,
        )
        return probe_result

    probe_weight[0].fill_(1)
    probe_weight[1].fill_(1)
    probe_weight[1, 0] = 2
    bf16_probe_token = token_id(ops.LLMM1(probe_weight, probe_hidden, 2).argmax())
    fp32_probe_tokens = {
        blocks: token_id(probe_candidate(blocks))
        for blocks in reduction_blocks
    }
    reference_probe_token = token_id(
        torch.mv(probe_weight.float(), probe_hidden[0].float()).argmax()
    )
    rounding_probe_pass = (
        bf16_probe_token == 0
        and all(token == 1 for token in fp32_probe_tokens.values())
        and reference_probe_token == 1
    )

    probe_weight.zero_()
    probe_hidden.zero_()
    zero_tie_pass = (
        token_id(ops.LLMM1(probe_weight, probe_hidden, 2).argmax()) == 0
        and all(
            token_id(probe_candidate(blocks)) == 0
            for blocks in reduction_blocks
        )
    )

    probe_hidden.fill_(float("nan"))
    nan_safe_pass = all(
        token_id(probe_candidate(blocks)) == 0 for blocks in reduction_blocks
    )

    random_reference_mismatches = dict.fromkeys(reduction_blocks, 0)
    probe_weight.normal_()
    for _ in range(3):
        probe_hidden.normal_()
        expected = token_id(
            torch.mv(probe_weight.float(), probe_hidden[0].float()).argmax()
        )
        for blocks in reduction_blocks:
            random_reference_mismatches[blocks] += (
                token_id(probe_candidate(blocks)) != expected
            )
    del probe_weight, probe_hidden
    torch.cuda.empty_cache()

    weight = torch.randn(args.m, args.k, device=device, dtype=dtype)
    hidden = torch.randn(1, args.k, device=device, dtype=dtype)
    local_values, local_indices, result = workspaces(args.m, device)

    def llmm1_floor() -> torch.Tensor:
        return ops.LLMM1(weight, hidden, 2)

    def baseline() -> torch.Tensor:
        return ops.LLMM1(weight, hidden, 2).float().argmax()

    def candidate(blocks: int) -> torch.Tensor:
        ops.LLMM1FP32Top1(
            weight, hidden, local_values, local_indices, result, blocks
        )
        return result

    baseline_token = token_id(baseline())
    llmm1_floor_ms = time_ms(llmm1_floor, args.warmup, args.iters)
    weight_bytes = weight.numel() * weight.element_size()

    print(
        f"device={properties.name}, compute_units={properties.multi_processor_count}, "
        f"shape=({args.m}, {args.k}), dtype={dtype}"
    )
    print(
        "rounding probe: "
        f"BF16={bf16_probe_token}, FP32={fp32_probe_tokens}, "
        f"reference={reference_probe_token}, pass={rounding_probe_pass}"
    )
    print(
        f"validation: zero_tie={zero_tie_pass}, "
        f"nan_safe={nan_safe_pass}, "
        f"random_reference_mismatches={random_reference_mismatches}"
    )
    print(f"LLMM1 floor={llmm1_floor_ms:.4f} ms")

    path = output_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for blocks in reduction_blocks:
        candidate_token = token_id(candidate(blocks))
        baseline_before_ms = time_ms(baseline, args.warmup, args.iters)
        candidate_ms = time_ms(
            lambda blocks=blocks: candidate(blocks), args.warmup, args.iters
        )
        baseline_after_ms = time_ms(baseline, args.warmup, args.iters)
        baseline_mean_ms = (baseline_before_ms + baseline_after_ms) / 2
        speedup = baseline_mean_ms / candidate_ms
        drift_pct = (baseline_after_ms / baseline_before_ms - 1) * 100
        candidate_gbps = weight_bytes / (candidate_ms * 1e6)
        validation_pass = (
            rounding_probe_pass
            and zero_tie_pass
            and nan_safe_pass
            and random_reference_mismatches[blocks] == 0
        )
        performance_pass = candidate_ms <= baseline_mean_ms * 0.995
        status = (
            "COMPONENT_PASS"
            if validation_pass and performance_pass
            else "STOP"
        )
        print(
            f"blocks={blocks}: B/C/B={baseline_before_ms:.4f}/"
            f"{candidate_ms:.4f}/{baseline_after_ms:.4f} ms, "
            f"speedup={speedup:.4f}x, candidate_bw={candidate_gbps:.1f} "
            f"GB/s, drift={drift_pct:+.2f}%, "
            f"tokens={baseline_token}/{candidate_token}, gate={status}"
        )
        rows.append(
            {
                "m": args.m,
                "k": args.k,
                "dtype": "bfloat16",
                "warmup": args.warmup,
                "iters": args.iters,
                "reduction_blocks": blocks,
                "llmm1_floor_ms": f"{llmm1_floor_ms:.6f}",
                "baseline_before_ms": f"{baseline_before_ms:.6f}",
                "candidate_ms": f"{candidate_ms:.6f}",
                "baseline_after_ms": f"{baseline_after_ms:.6f}",
                "baseline_mean_ms": f"{baseline_mean_ms:.6f}",
                "baseline_drift_pct": f"{drift_pct:+.4f}",
                "speedup": f"{speedup:.6f}",
                "candidate_weight_gbps": f"{candidate_gbps:.4f}",
                "baseline_token_id": baseline_token,
                "candidate_token_id": candidate_token,
                "rounding_probe_bf16_token": bf16_probe_token,
                "rounding_probe_fp32_token": fp32_probe_tokens[blocks],
                "rounding_probe_reference_token": reference_probe_token,
                "rounding_probe_pass": rounding_probe_pass,
                "zero_tie_pass": zero_tie_pass,
                "nan_safe_pass": nan_safe_pass,
                "random_reference_mismatches": (
                    random_reference_mismatches[blocks]
                ),
                "validation_pass": validation_pass,
                "performance_pass": performance_pass,
                "status": status,
            }
        )

    best = max(rows, key=lambda row: float(row["speedup"]))
    print(
        f"best blocks={best['reduction_blocks']}, "
        f"speedup={best['speedup']}x, gate={best['status']}"
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
