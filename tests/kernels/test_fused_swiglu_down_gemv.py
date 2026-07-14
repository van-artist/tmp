# SPDX-License-Identifier: Apache-2.0
"""Correctness and opt-in timing for Qwen3.5 BF16 decode MLP fusions."""

import os

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.qwen35_fused_mlp import (
    qwen35_gate_up_swiglu_down,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Qwen3.5 fusion candidates require ROCm"
)


def _inputs(seed: int = 13):
    set_random_seed(seed)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    x = torch.randn((1, HIDDEN_SIZE), device=device, dtype=dtype) * 0.1
    gate_up_weight = torch.randn(
        (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE), device=device, dtype=dtype
    ) * 0.01
    down_weight = torch.randn(
        (HIDDEN_SIZE, INTERMEDIATE_SIZE), device=device, dtype=dtype
    ) * 0.01
    return x, gate_up_weight, down_weight


def _baseline(x, gate_up_weight, down_weight):
    gate_up = F.linear(x, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    activation = F.silu(gate) * up
    return activation, F.linear(activation, down_weight)


@pytest.mark.parametrize("seed", [1, 13])
@torch.inference_mode()
def test_gate_up_swiglu_matches_reference(seed: int):
    from vllm.platforms.rocm import on_gfx936

    if not on_gfx936():
        pytest.skip("exact candidates are intended for gfx936")
    x, gate_up_weight, down_weight = _inputs(seed)
    reference_activation, _ = _baseline(x, gate_up_weight, down_weight)
    actual = torch.ops.vllm.qwen35_gate_up_swiglu(x, gate_up_weight)
    torch.testing.assert_close(actual, reference_activation, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("seed", [1, 13])
@torch.inference_mode()
def test_swiglu_down_matches_reference(seed: int):
    from vllm.platforms.rocm import on_gfx936

    if not on_gfx936():
        pytest.skip("exact candidates are intended for gfx936")
    x, gate_up_weight, down_weight = _inputs(seed)
    gate_up = F.linear(x, gate_up_weight)
    _, reference = _baseline(x, gate_up_weight, down_weight)
    actual = torch.ops.vllm.qwen35_swiglu_down(gate_up, down_weight)
    torch.testing.assert_close(actual, reference, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("seed", [1, 13])
@torch.inference_mode()
def test_two_kernel_full_mlp_matches_reference(seed: int):
    from vllm.platforms.rocm import on_gfx936

    if not on_gfx936():
        pytest.skip("exact candidates are intended for gfx936")
    x, gate_up_weight, down_weight = _inputs(seed)
    _, reference = _baseline(x, gate_up_weight, down_weight)
    actual = qwen35_gate_up_swiglu_down(x, gate_up_weight, down_weight)
    torch.testing.assert_close(actual, reference, atol=5e-2, rtol=2e-2)


@pytest.mark.skipif(
    os.getenv("VLLM_RUN_QWEN35_MLP_BENCHMARK") != "1",
    reason="set VLLM_RUN_QWEN35_MLP_BENCHMARK=1 to benchmark",
)
@torch.inference_mode()
def test_benchmark_qwen35_mlp_candidates():
    from triton.testing import do_bench

    from vllm.platforms.rocm import on_gfx936

    if not on_gfx936():
        pytest.skip("exact candidates are intended for gfx936")
    x, gate_up_weight, down_weight = _inputs()

    def baseline():
        return _baseline(x, gate_up_weight, down_weight)[1]

    def fused_gate_up():
        activation = torch.ops.vllm.qwen35_gate_up_swiglu(x, gate_up_weight)
        return F.linear(activation, down_weight)

    def fused_down():
        return qwen35_gate_up_swiglu_down(x, gate_up_weight, down_weight)

    baseline_ms = do_bench(baseline)
    gate_up_ms = do_bench(fused_gate_up)
    down_ms = do_bench(fused_down)
    print(f"baseline_3_kernel={baseline_ms * 1000:.2f} us")
    print(
        f"gate_up_swiglu_plus_down={gate_up_ms * 1000:.2f} us "
        f"speedup={baseline_ms / gate_up_ms:.3f}x"
    )
    print(
        f"gate_up_plus_swiglu_down={down_ms * 1000:.2f} us "
        f"speedup={baseline_ms / down_ms:.3f}x"
    )
