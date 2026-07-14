# SPDX-License-Identifier: Apache-2.0
"""Standalone Qwen3.5 decode GEMV backend and padding benchmark.

This script intentionally avoids pytest so it can run in the contest container
without installing test-only dependencies.  It measures one weight at a time
and never includes tensor construction or padding copies in timed regions.

Examples:

    python tests/kernels/run_qwen35_decode_gemv_arena.py
    python tests/kernels/run_qwen35_decode_gemv_arena.py --shapes down gate_up
    python tests/kernels/run_qwen35_decode_gemv_arena.py --shapes all
    python tests/kernels/run_qwen35_decode_gemv_arena.py \
        --shapes down --down-padded-k 17408 17472 17536 17664 17920
"""

from __future__ import annotations

import argparse
import gc
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from triton.testing import do_bench


@dataclass(frozen=True)
class GemvShape:
    name: str
    out_features: int
    in_features: int


SHAPES = {
    "down": GemvShape("down", 5120, 17408),
    "gate_up": GemvShape("gate_up", 34816, 5120),
    "gdn_out": GemvShape("gdn_out", 5120, 6144),
    "gdn_packed": GemvShape("gdn_packed", 16480, 5120),
    "full_qkv": GemvShape("full_qkv", 14336, 5120),
    "lm_head": GemvShape("lm_head", 248320, 5120),
}


def _set_blas_library(name: str | object) -> None:
    # PyTorch retains CUDA-compatible naming for this API on ROCm/DTK.
    torch.backends.cuda.preferred_blas_library(name)


def _make_tensors(
    shape: GemvShape,
    *,
    padded_k: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    logical_k = shape.in_features
    physical_k = logical_k if padded_k is None else padded_k
    if physical_k < logical_k:
        raise ValueError(
            f"padded K {physical_k} is smaller than logical K {logical_k}"
        )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    # Scale inputs so the BF16 reference remains in a compact numerical range.
    x = torch.randn(
        (1, physical_k),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ) / math.sqrt(logical_k)
    weight = torch.randn(
        (shape.out_features, physical_k),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    if physical_k != logical_k:
        x[:, logical_k:].zero_()
        weight[:, logical_k:].zero_()
    return x.contiguous(), weight.contiguous(), physical_k


def _torch_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight)


def _production_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    return rocm_unquantized_gemm_impl(x, weight)


def _bench(
    fn: Callable[[], torch.Tensor],
    *,
    warmup_ms: int,
    rep_ms: int,
) -> float:
    # Compile/initialize the selected BLAS solution before timing.
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    return float(do_bench(fn, warmup=warmup_ms, rep=rep_ms))


def _compare(
    result: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float]:
    diff = (result.float() - reference.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    # This is a screening tolerance, not the final model accuracy gate.
    torch.testing.assert_close(result, reference, atol=0.125, rtol=0.02)
    return max_abs, mean_abs


def _print_result(
    *,
    label: str,
    elapsed_ms: float,
    logical_bytes: int,
    physical_bytes: int,
    max_abs: float,
    mean_abs: float,
    baseline_ms: float,
) -> None:
    seconds = elapsed_ms / 1000.0
    useful_gbps = logical_bytes / seconds / 1.0e9
    physical_gbps = physical_bytes / seconds / 1.0e9
    print(
        f"  {label:<27} {elapsed_ms:8.3f} ms  "
        f"speedup={baseline_ms / elapsed_ms:6.3f}x  "
        f"useful={useful_gbps:7.1f} GB/s  "
        f"physical={physical_gbps:7.1f} GB/s  "
        f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}"
    )


def run_case(
    shape: GemvShape,
    *,
    padded_k: int | None,
    warmup_ms: int,
    rep_ms: int,
    seed: int,
) -> None:
    x, weight, physical_k = _make_tensors(shape, padded_k=padded_k, seed=seed)
    logical_bytes = shape.out_features * shape.in_features * weight.element_size()
    physical_bytes = shape.out_features * physical_k * weight.element_size()
    suffix = "" if physical_k == shape.in_features else f" padded_k={physical_k}"
    print(
        f"\ncase={shape.name}{suffix} W=[{shape.out_features},{physical_k}] "
        f"physical={physical_bytes / 2**20:.1f} MiB"
    )

    previous_blas = torch.backends.cuda.preferred_blas_library()
    try:
        _set_blas_library("cublas")
        reference = _torch_linear(x, weight)
        torch.cuda.synchronize()
        baseline_ms = _bench(
            lambda: _torch_linear(x, weight),
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )
        _print_result(
            label="hipblas",
            elapsed_ms=baseline_ms,
            logical_bytes=logical_bytes,
            physical_bytes=physical_bytes,
            max_abs=0.0,
            mean_abs=0.0,
            baseline_ms=baseline_ms,
        )

        _set_blas_library("cublaslt")
        result = _torch_linear(x, weight)
        torch.cuda.synchronize()
        max_abs, mean_abs = _compare(result, reference)
        elapsed_ms = _bench(
            lambda: _torch_linear(x, weight),
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )
        _print_result(
            label="hipblaslt",
            elapsed_ms=elapsed_ms,
            logical_bytes=logical_bytes,
            physical_bytes=physical_bytes,
            max_abs=max_abs,
            mean_abs=mean_abs,
            baseline_ms=baseline_ms,
        )

        # Restore the service's normal preference before exercising its actual
        # dispatch.  This may select the shape-specific AITER path.
        _set_blas_library(previous_blas)
        result = _production_linear(x, weight)
        torch.cuda.synchronize()
        max_abs, mean_abs = _compare(result, reference)
        elapsed_ms = _bench(
            lambda: _production_linear(x, weight),
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )
        _print_result(
            label="production_dispatch",
            elapsed_ms=elapsed_ms,
            logical_bytes=logical_bytes,
            physical_bytes=physical_bytes,
            max_abs=max_abs,
            mean_abs=mean_abs,
            baseline_ms=baseline_ms,
        )
    finally:
        _set_blas_library(previous_blas)
        del x, weight
        if "reference" in locals():
            del reference
        if "result" in locals():
            del result
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["down"],
        choices=[*SHAPES, "all"],
        help="GEMV shapes to benchmark; default: down",
    )
    parser.add_argument(
        "--down-padded-k",
        nargs="+",
        type=int,
        default=[17408, 17472, 17536, 17664, 17920],
        help="Physical K values tested for the Down projection",
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is not available")
    props = torch.cuda.get_device_properties(0)
    arch = str(getattr(props, "gcnArchName", "unknown"))
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"arch={arch}")
    if arch.split(":", 1)[0] != "gfx936":
        raise RuntimeError(f"expected gfx936, got {arch}")
    print(
        "times exclude allocation/padding; bandwidth counts weight bytes only; "
        "production dispatch obeys current VLLM_ROCM_* environment variables"
    )

    selected = list(SHAPES) if "all" in args.shapes else args.shapes
    for name in selected:
        shape = SHAPES[name]
        if name == "down":
            for padded_k in args.down_padded_k:
                run_case(
                    shape,
                    padded_k=padded_k,
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                    seed=args.seed,
                )
        else:
            run_case(
                shape,
                padded_k=None,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()
