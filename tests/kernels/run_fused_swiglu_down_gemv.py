# SPDX-License-Identifier: Apache-2.0
"""Standalone Qwen3.5 BF16 MLP fusion correctness/performance runner.

This file deliberately does not use pytest, so it can run in a minimal vLLM
container without installing tblib or the repository's test requirements.
"""

import argparse

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from vllm.model_executor.layers.qwen35_fused_mlp import (
    qwen35_gate_up_swiglu_down,
)

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408


def make_inputs(seed: int):
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    x = torch.randn((1, HIDDEN_SIZE), device=device, dtype=dtype) * 0.1
    gate_up_weight = torch.randn(
        (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device=device,
        dtype=dtype,
    ) * 0.01
    down_weight = torch.randn(
        (HIDDEN_SIZE, INTERMEDIATE_SIZE),
        device=device,
        dtype=dtype,
    ) * 0.01
    return x, gate_up_weight, down_weight


def baseline(x, gate_up_weight, down_weight):
    gate_up = F.linear(x, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    activation = F.silu(gate) * up
    return activation, F.linear(activation, down_weight)


def check_close(name: str, actual: torch.Tensor, expected: torch.Tensor):
    diff = (actual.float() - expected.float()).abs()
    print(
        f"{name}: max_abs={diff.max().item():.8f}, "
        f"mean_abs={diff.mean().item():.8f}"
    )
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="skip timing after checking numerical results",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is not available")
    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", "unknown"))
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"arch={arch}")
    if arch.split(":", 1)[0] != "gfx936":
        raise RuntimeError(f"expected gfx936, got {arch}")

    print("allocating exact Qwen3.5-27B decode tensors...")
    x, gate_up_weight, down_weight = make_inputs(args.seed)

    print("JIT compiling and checking correctness...")
    reference_activation, reference_output = baseline(
        x, gate_up_weight, down_weight
    )
    fused_activation = torch.ops.vllm.qwen35_gate_up_swiglu(
        x, gate_up_weight
    )
    fused_gate_up_output = F.linear(fused_activation, down_weight)
    fused_down_output = qwen35_gate_up_swiglu_down(
        x, gate_up_weight, down_weight
    )
    torch.cuda.synchronize()

    check_close("gate_up+swiglu activation", fused_activation, reference_activation)
    check_close(
        "gate_up+swiglu then down", fused_gate_up_output, reference_output
    )
    check_close("gate_up then swiglu+down", fused_down_output, reference_output)
    print("correctness=PASS")

    if args.correctness_only:
        return

    def run_baseline():
        return baseline(x, gate_up_weight, down_weight)[1]

    def run_fused_gate_up():
        activation = torch.ops.vllm.qwen35_gate_up_swiglu(
            x, gate_up_weight
        )
        return F.linear(activation, down_weight)

    def run_fused_down():
        return qwen35_gate_up_swiglu_down(x, gate_up_weight, down_weight)

    print("benchmarking warmed kernels...")
    baseline_ms = do_bench(run_baseline)
    gate_up_ms = do_bench(run_fused_gate_up)
    down_ms = do_bench(run_fused_down)
    print(f"baseline_3_kernel={baseline_ms * 1000:.2f} us")
    print(
        f"gate_up_swiglu_plus_down={gate_up_ms * 1000:.2f} us, "
        f"speedup={baseline_ms / gate_up_ms:.3f}x"
    )
    print(
        f"gate_up_plus_swiglu_down={down_ms * 1000:.2f} us, "
        f"speedup={baseline_ms / down_ms:.3f}x"
    )


if __name__ == "__main__":
    main()
