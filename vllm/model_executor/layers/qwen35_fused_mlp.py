# SPDX-License-Identifier: Apache-2.0
"""Experimental BF16 Qwen3.5 decode MLP fusion kernels for gfx936 tuning.

These kernels intentionally are not wired into the model.  They provide
standalone candidates for correctness and device benchmarking before changing
the established inference path.
"""

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_HIDDEN_SIZE = 5120
_INTERMEDIATE_SIZE = 17408


@triton.jit
def _qwen35_gate_up_swiglu_kernel(
    x_ptr,
    gate_up_weight_ptr,
    out_ptr,
    stride_weight_row,
    INPUT_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one paired gate/up row and its SwiGLU activation."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    gate_acc = tl.zeros((), dtype=tl.float32)
    up_acc = tl.zeros((), dtype=tl.float32)

    for start in range(0, INPUT_SIZE, BLOCK_K):
        cols = start + offsets
        mask = cols < INPUT_SIZE
        x = tl.load(x_ptr + cols, mask=mask, other=0.0)
        gate = tl.load(
            gate_up_weight_ptr + row * stride_weight_row + cols,
            mask=mask,
            other=0.0,
        )
        up = tl.load(
            gate_up_weight_ptr
            + (INTERMEDIATE_SIZE + row) * stride_weight_row
            + cols,
            mask=mask,
            other=0.0,
        )
        gate_acc += tl.sum(x.to(tl.float32) * gate.to(tl.float32), axis=0)
        up_acc += tl.sum(x.to(tl.float32) * up.to(tl.float32), axis=0)

    # Preserve the BF16 boundary of the baseline GEMV before its activation.
    gate = gate_acc.to(tl.bfloat16).to(tl.float32)
    up = up_acc.to(tl.bfloat16).to(tl.float32)
    activated = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + row, activated.to(tl.bfloat16))


@triton.jit
def _qwen35_swiglu_down_kernel(
    gate_up_ptr,
    down_weight_ptr,
    out_ptr,
    stride_down_row,
    HIDDEN_SIZE: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fuse SwiGLU with a tiled Down GEMV without an activation tensor."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((1, BLOCK_M), dtype=tl.float32)

    for start in range(0, INTERMEDIATE_SIZE, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        col_mask = cols < INTERMEDIATE_SIZE
        gate = tl.load(gate_up_ptr + cols, mask=col_mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(
            gate_up_ptr + INTERMEDIATE_SIZE + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        # Match the standalone activation's BF16 output before Down GEMV.
        activated = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16)
        weights = tl.load(
            down_weight_ptr
            + rows[:, None] * stride_down_row
            + cols[None, :],
            mask=(rows[:, None] < HIDDEN_SIZE) & col_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(
            activated[None, :],
            tl.trans(weights),
            input_precision="ieee",
        )

    tl.store(
        out_ptr + rows,
        acc[0, :].to(tl.bfloat16),
        mask=rows < HIDDEN_SIZE,
    )


def _validate_gate_up_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    if x.shape != (1, _HIDDEN_SIZE):
        raise ValueError(f"expected x shape (1, {_HIDDEN_SIZE}), got {x.shape}")
    if weight.shape != (2 * _INTERMEDIATE_SIZE, _HIDDEN_SIZE):
        raise ValueError("unexpected Qwen3.5 gate/up weight shape")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("x and weight must be BF16")
    if (
        x.device != weight.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("x and weight must be contiguous on the same device")


def qwen35_gate_up_swiglu_impl(
    x: torch.Tensor, gate_up_weight: torch.Tensor
) -> torch.Tensor:
    """Fused one-token Gate-Up GEMV and SwiGLU candidate."""
    _validate_gate_up_inputs(x, gate_up_weight)
    out = x.new_empty((1, _INTERMEDIATE_SIZE))
    _qwen35_gate_up_swiglu_kernel[(_INTERMEDIATE_SIZE,)](
        x,
        gate_up_weight,
        out,
        gate_up_weight.stride(0),
        INPUT_SIZE=_HIDDEN_SIZE,
        INTERMEDIATE_SIZE=_INTERMEDIATE_SIZE,
        BLOCK_K=256,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    return out


def qwen35_gate_up_swiglu_fake(
    x: torch.Tensor, gate_up_weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((1, gate_up_weight.shape[0] // 2))


direct_register_custom_op(
    op_name="qwen35_gate_up_swiglu",
    op_func=qwen35_gate_up_swiglu_impl,
    fake_impl=qwen35_gate_up_swiglu_fake,
)


def qwen35_swiglu_down_impl(
    gate_up: torch.Tensor, down_weight: torch.Tensor
) -> torch.Tensor:
    """Fused SwiGLU and one-token Down GEMV candidate."""
    if gate_up.shape != (1, 2 * _INTERMEDIATE_SIZE):
        raise ValueError("unexpected Qwen3.5 gate/up activation shape")
    if down_weight.shape != (_HIDDEN_SIZE, _INTERMEDIATE_SIZE):
        raise ValueError("unexpected Qwen3.5 down weight shape")
    if gate_up.dtype != torch.bfloat16 or down_weight.dtype != torch.bfloat16:
        raise ValueError("gate_up and down_weight must be BF16")
    if (
        gate_up.device != down_weight.device
        or not gate_up.is_contiguous()
        or not down_weight.is_contiguous()
    ):
        raise ValueError("inputs must be contiguous on the same device")

    out = gate_up.new_empty((1, _HIDDEN_SIZE))
    _qwen35_swiglu_down_kernel[(triton.cdiv(_HIDDEN_SIZE, 16),)](
        gate_up,
        down_weight,
        out,
        down_weight.stride(0),
        HIDDEN_SIZE=_HIDDEN_SIZE,
        INTERMEDIATE_SIZE=_INTERMEDIATE_SIZE,
        BLOCK_M=16,
        BLOCK_K=64,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
        matrix_instr_nonkdim=16,
        kpack=2,
    )
    return out


def qwen35_swiglu_down_fake(
    gate_up: torch.Tensor, down_weight: torch.Tensor
) -> torch.Tensor:
    return gate_up.new_empty((1, down_weight.shape[0]))


direct_register_custom_op(
    op_name="qwen35_swiglu_down",
    op_func=qwen35_swiglu_down_impl,
    fake_impl=qwen35_swiglu_down_fake,
)


def qwen35_gate_up_swiglu_down(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Two-kernel full MLP candidate: Gate-Up GEMV, then SwiGLU+Down."""
    _validate_gate_up_inputs(x, gate_up_weight)
    gate_up = F.linear(x, gate_up_weight)
    return torch.ops.vllm.qwen35_swiglu_down(gate_up, down_weight)
