# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the Qwen3.5 fused SwiGLU decode path."""

import pytest
import torch
import torch.nn.functional as F

# Importing the module registers torch.ops.vllm.qwen35_fused_gate_up_swiglu.
from vllm.model_executor.layers import fused_swiglu  # noqa: F401
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408


def _reference_swiglu(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
) -> torch.Tensor:
    gate_up = F.linear(x, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Qwen3.5 fused SwiGLU GEMV is a gfx936 ROCm specialization",
)
@pytest.mark.parametrize("seed", [1, 13])
@torch.inference_mode()
def test_fused_swiglu_down_gemv_matches_reference(seed: int):
    """Check both the fused activation and its downstream GEMV result."""
    from vllm.platforms.rocm import on_gfx936

    if not on_gfx936():
        pytest.skip("exact Qwen3.5 SwiGLU kernel is enabled only on gfx936")

    set_random_seed(seed)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # Keep values close to model initialization ranges. This avoids oversized
    # random outputs obscuring errors when the down projection accumulates the
    # activation differences.
    x = torch.randn((1, HIDDEN_SIZE), device=device, dtype=dtype) * 0.1
    gate_up_weight = (
        torch.randn(
            (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            device=device,
            dtype=dtype,
        )
        * 0.01
    )
    down_weight = (
        torch.randn(
            (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            device=device,
            dtype=dtype,
        )
        * 0.01
    )

    reference_activation = _reference_swiglu(x, gate_up_weight)
    fused_activation = torch.ops.vllm.qwen35_fused_gate_up_swiglu(
        x, gate_up_weight
    )

    torch.testing.assert_close(
        fused_activation,
        reference_activation,
        atol=2e-2,
        rtol=2e-2,
    )

    reference_output = F.linear(reference_activation, down_weight)
    fused_output = F.linear(fused_activation, down_weight)
    torch.testing.assert_close(
        fused_output,
        reference_output,
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Qwen3.5 fused SwiGLU GEMV is a gfx936 ROCm specialization",
)
@torch.inference_mode()
def test_fused_swiglu_zero_input():
    """Zero input must produce an exact zero SwiGLU activation."""
    from vllm.platforms.rocm import on_gfx936

    if not on_gfx936():
        pytest.skip("exact Qwen3.5 SwiGLU kernel is enabled only on gfx936")

    x = torch.zeros(
        (1, HIDDEN_SIZE), device="cuda:0", dtype=torch.bfloat16
    )
    gate_up_weight = torch.randn(
        (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    actual = torch.ops.vllm.qwen35_fused_gate_up_swiglu(x, gate_up_weight)

    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
