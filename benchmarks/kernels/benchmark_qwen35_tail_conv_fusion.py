"""Validate/timing the exact single-prefill GDN convolution tail fusion.

The reference is the production three-launch tail route: causal convolution,
post-convolution preparation, then the 64-token local gate cumsum.  Candidate
outputs and the indexed convolution state must be bitwise identical.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
from vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv import (
    fused_post_conv_prep,
)
from vllm.model_executor.layers.fla.ops.fused_qwen35_prefill_conv import (
    fused_qwen35_prefill_conv_prep,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.platforms.rocm import on_gfx936
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata


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
    return seed.clone().transpose(1, 2)


def run_case(
    length: int, has_initial: bool, warmup: int, repeats: int
) -> dict[str, object]:
    torch.manual_seed(1701 + length + int(has_initial))
    device = torch.device("cuda")
    packed = torch.randn(
        (length, PACKED), device=device, dtype=torch.bfloat16
    ) * 0.1
    mixed_qkv = packed[:, :QKV]
    b = packed[:, -2 * HV : -HV]
    a = packed[:, -HV:]
    weight = torch.randn(
        (QKV, WIDTH), device=device, dtype=torch.bfloat16
    ) * 0.1
    state_seed = torch.randn(
        (1, WIDTH - 1, QKV), device=device, dtype=torch.bfloat16
    ) * 0.1
    state_indices = torch.zeros((1,), device=device, dtype=torch.int32)
    has_initial_tensor = torch.tensor(
        [has_initial], device=device, dtype=torch.bool
    )
    query_start = torch.tensor(
        [0, length], device=device, dtype=torch.int32
    )
    A_log = torch.randn((HV,), device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn((HV,), device=device, dtype=torch.float32) * 0.1
    nums_dict, batch_ptr, token_chunk_offset_ptr = (
        compute_causal_conv1d_metadata(
            torch.tensor([0, length], dtype=torch.int32), device=device
        )
    )
    metadata = SimpleNamespace(
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )

    def reference(state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        conv_output = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            weight,
            None,
            state,
            query_start,
            cache_indices=state_indices,
            has_initial_state=has_initial_tensor,
            activation="silu",
            metadata=metadata,
        ).transpose(0, 1)
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=conv_output,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            num_k_heads=H,
            head_k_dim=K,
            head_v_dim=V,
            apply_l2norm=True,
            output_g_exp=False,
            output_local_cumsum=False,
        )
        g = chunk_local_cumsum(g.unsqueeze(0), chunk_size=64).squeeze(0)
        return q, k, v, g, beta

    def candidate(state: torch.Tensor) -> tuple[torch.Tensor, ...]:
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
        )

    state_ref = make_state(state_seed)
    state_got = make_state(state_seed)
    ref = reference(state_ref)
    got = candidate(state_got)
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
            f"tail fusion is not bitwise exact: {length=}, {has_initial=}, "
            f"{equality=}, {max_abs=}"
        )

    timing_ref_state = make_state(state_seed)
    timing_got_state = make_state(state_seed)
    ref_ms = median_ms(
        lambda: reference(timing_ref_state), warmup=warmup, repeats=repeats
    )
    got_ms = median_ms(
        lambda: candidate(timing_got_state), warmup=warmup, repeats=repeats
    )
    return {
        "T": length,
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
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[94, 1674, 2231, 3478]
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not on_gfx936():
        raise RuntimeError("benchmark requires the competition gfx936 GPU")
    result = {
        "removed_launches_per_48_layers": 96,
        "cases": [
            run_case(length, has_initial, args.warmup, args.repeats)
            for length in args.lengths
            for has_initial in (False, True)
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
