# SPDX-License-Identifier: Apache-2.0
"""Exact-shape fused gate/up projection and SwiGLU for Qwen3.5 decode."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _qwen35_fused_gate_up_swiglu_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_weight_row,
    INPUT_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute paired gate/up rows and apply SwiGLU before storing."""
    intermediate_row = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    gate_acc = tl.zeros((), dtype=tl.float32)
    up_acc = tl.zeros((), dtype=tl.float32)

    for k_start in range(0, INPUT_SIZE, BLOCK_K):
        k = k_start + offs_k
        mask = k < INPUT_SIZE
        x = tl.load(x_ptr + k, mask=mask, other=0.0).to(tl.float32)
        gate_weight = tl.load(
            weight_ptr + intermediate_row * stride_weight_row + k,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up_weight = tl.load(
            weight_ptr
            + (INTERMEDIATE_SIZE + intermediate_row) * stride_weight_row
            + k,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate_acc += tl.sum(x * gate_weight, axis=0)
        up_acc += tl.sum(x * up_weight, axis=0)

    # Match the standalone BF16 gate_up projection before SiluAndMul applies
    # its FP32 transcendental and stores a BF16 activated value.
    gate = gate_acc.to(x_ptr.dtype.element_ty).to(tl.float32)
    up = up_acc.to(x_ptr.dtype.element_ty)
    silu = (gate * tl.sigmoid(gate)).to(x_ptr.dtype.element_ty)
    activated = (silu * up).to(x_ptr.dtype.element_ty)
    tl.store(out_ptr + intermediate_row, activated)


def qwen35_fused_gate_up_swiglu_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Project one Qwen3.5 decode row and return its SwiGLU activation."""
    if x.shape != (1, 5120):
        raise ValueError(f"expected x shape (1, 5120), got {x.shape}")
    if weight.shape != (34816, 5120):
        raise ValueError(f"expected weight shape (34816, 5120), got {weight.shape}")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("x and weight must be BF16")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device")

    out = torch.empty((1, 17408), dtype=x.dtype, device=x.device)
    _qwen35_fused_gate_up_swiglu_kernel[(17408,)](
        x,
        weight,
        out,
        weight.stride(0),
        INPUT_SIZE=5120,
        INTERMEDIATE_SIZE=17408,
        BLOCK_K=256,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    return out


def qwen35_fused_gate_up_swiglu_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((1, weight.shape[0] // 2))


direct_register_custom_op(
    op_name="qwen35_fused_gate_up_swiglu",
    op_func=qwen35_fused_gate_up_swiglu_impl,
    fake_impl=qwen35_fused_gate_up_swiglu_fake,
)
