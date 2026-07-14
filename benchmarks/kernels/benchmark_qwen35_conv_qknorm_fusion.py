"""Validate and time Q/K norm inside the exact Qwen3.5 GDN conv launch.

Run on the competition gfx936 worker after installing or overlaying this tree::

    python benchmarks/kernels/benchmark_qwen35_conv_qknorm_fusion.py \
        --warmup 12 --repeats 51

The reference and candidate use the same production kernel.  The reference
retains the standalone in-place Q/K-normalization launch; the candidate keeps
the causal-convolution BF16 store/load rounding boundary inside the convolution
launch.  Every output and the complete convolution cache must be bitwise equal
before latency is reported.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from vllm.model_executor.layers.fla.ops.fused_qwen35_prefill_conv import (
    fused_qwen35_prefill_conv_prep,
)
from vllm.platforms.rocm import on_gfx936


T = 4096
H = 16
HV = 48
K = 128
V = 128
WIDTH = 4
QKV = 2 * H * K + HV * V
PACKED = 16480


def median_ms(fn: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def make_state(seed: torch.Tensor) -> torch.Tensor:
    # Production state is a transpose view: [slot, channel, history] with
    # channel stride 1 and history stride QKV.
    return seed.clone().transpose(1, 2)


def run_case(has_initial: bool, warmup: int, repeats: int) -> dict[str, object]:
    torch.manual_seed(1234 + int(has_initial))
    device = torch.device("cuda")
    packed = torch.randn((T, PACKED), device=device, dtype=torch.bfloat16) * 0.1
    mixed_qkv = packed[:, :QKV]
    b = packed[:, -2 * HV : -HV]
    a = packed[:, -HV:]
    weight = torch.randn((QKV, WIDTH), device=device, dtype=torch.bfloat16) * 0.1
    state_seed = torch.randn(
        (1, WIDTH - 1, QKV), device=device, dtype=torch.bfloat16
    ) * 0.1
    state_indices = torch.zeros((1,), device=device, dtype=torch.int32)
    has_initial_tensor = torch.tensor(
        [has_initial], device=device, dtype=torch.bool
    )
    A_log = torch.randn((HV,), device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn((HV,), device=device, dtype=torch.float32) * 0.1

    def invoke(state: torch.Tensor, fused: bool) -> tuple[torch.Tensor, ...]:
        return fused_qwen35_prefill_conv_prep(
            mixed_qkv=mixed_qkv,
            conv_weight=weight,
            conv_state=state,
            state_indices=state_indices,
            has_initial_state=has_initial_tensor,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            fuse_qk_norm=fused,
        )

    state_ref = make_state(state_seed)
    state_got = make_state(state_seed)
    ref = invoke(state_ref, False)
    got = invoke(state_got, True)
    torch.cuda.synchronize()

    names = ("q", "k", "v", "g", "beta")
    equality = {name: torch.equal(x, y) for name, x, y in zip(names, ref, got)}
    max_abs = {
        name: float((x.float() - y.float()).abs().max().item())
        for name, x, y in zip(names, ref, got)
    }
    equality["conv_state"] = torch.equal(state_ref, state_got)
    max_abs["conv_state"] = float(
        (state_ref.float() - state_got.float()).abs().max().item()
    )
    if not all(equality.values()):
        raise AssertionError(
            f"Q/K-in-conv fusion is not bitwise exact: {equality=}, {max_abs=}"
        )

    # State writes are deterministic and have the same byte volume on every
    # iteration, so repeated calls are suitable for the latency comparison.
    timing_state_ref = make_state(state_seed)
    timing_state_got = make_state(state_seed)
    ref_ms = median_ms(
        lambda: invoke(timing_state_ref, False), warmup=warmup, repeats=repeats
    )
    got_ms = median_ms(
        lambda: invoke(timing_state_got, True), warmup=warmup, repeats=repeats
    )
    return {
        "has_initial_state": has_initial,
        "bitwise": equality,
        "max_abs": max_abs,
        "reference_ms": ref_ms,
        "candidate_ms": got_ms,
        "speedup": ref_ms / got_ms,
        "saving_ms": ref_ms - got_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=51)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires the target ROCm GPU")
    if not on_gfx936():
        raise RuntimeError("This prototype is guarded for gfx936 only")

    result = {
        "shape": {"T": T, "H": H, "HV": HV, "K": K, "V": V},
        "removed_launches_per_48_layers": 48,
        "cases": [
            run_case(False, args.warmup, args.repeats),
            run_case(True, args.warmup, args.repeats),
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
