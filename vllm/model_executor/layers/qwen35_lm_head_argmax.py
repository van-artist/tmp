# SPDX-License-Identifier: Apache-2.0
"""Experimental Qwen3.5 BF16 LM-head GEMV with online greedy argmax."""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_HIDDEN_SIZE = 5120


@triton.jit
def _lm_head_partial_argmax_kernel(
    x_ptr,
    weight_ptr,
    partial_values_ptr,
    partial_indices_ptr,
    stride_weight_row,
    VOCAB_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute a vocab-row tile and emit only its maximum value/index."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start in range(0, HIDDEN_SIZE, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        col_mask = cols < HIDDEN_SIZE
        x = tl.load(x_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        weights = tl.load(
            weight_ptr + rows[:, None] * stride_weight_row + cols[None, :],
            mask=(rows[:, None] < VOCAB_SIZE) & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(weights * x[None, :], axis=1)

    # The normal LM head stores BF16 logits before Sampler converts them to
    # FP32.  Round here before comparing so the greedy ordering has the same
    # precision boundary.
    logits = acc.to(tl.bfloat16).to(tl.float32)
    logits = tl.where(rows < VOCAB_SIZE, logits, -float("inf"))
    local_value, local_offset = tl.max(logits, axis=0, return_indices=True)
    partial = tl.program_id(0)
    tl.store(partial_values_ptr + partial, local_value)
    tl.store(
        partial_indices_ptr + partial,
        partial * BLOCK_M + local_offset,
    )


@triton.jit
def _lm_head_finalize_argmax_kernel(
    partial_values_ptr,
    partial_indices_ptr,
    output_ptr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_PARTIALS)
    mask = offsets < NUM_PARTIALS
    values = tl.load(
        partial_values_ptr + offsets, mask=mask, other=-float("inf")
    )
    _, winner = tl.max(values, axis=0, return_indices=True)
    token = tl.load(partial_indices_ptr + winner)
    tl.store(output_ptr, token)


def qwen35_lm_head_argmax_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Return one greedy token without materializing full-vocabulary logits."""
    if x.shape != (1, _HIDDEN_SIZE):
        raise ValueError(f"expected x shape (1, {_HIDDEN_SIZE}), got {x.shape}")
    if weight.ndim != 2 or weight.shape[1] != _HIDDEN_SIZE:
        raise ValueError("unexpected LM-head weight shape")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("x and weight must be BF16")
    if (
        x.device != weight.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("x and weight must be contiguous on the same device")

    vocab_size = weight.shape[0]
    block_m = 16
    num_partials = triton.cdiv(vocab_size, block_m)
    partial_values = torch.empty(
        (num_partials,), device=x.device, dtype=torch.float32
    )
    partial_indices = torch.empty(
        (num_partials,), device=x.device, dtype=torch.int32
    )
    output = torch.empty((1,), device=x.device, dtype=torch.int32)
    _lm_head_partial_argmax_kernel[(num_partials,)](
        x,
        weight,
        partial_values,
        partial_indices,
        weight.stride(0),
        VOCAB_SIZE=vocab_size,
        HIDDEN_SIZE=_HIDDEN_SIZE,
        BLOCK_M=block_m,
        BLOCK_K=256,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    block_partials = triton.next_power_of_2(num_partials)
    _lm_head_finalize_argmax_kernel[(1,)](
        partial_values,
        partial_indices,
        output,
        NUM_PARTIALS=num_partials,
        BLOCK_PARTIALS=block_partials,
        num_warps=8,
        num_stages=1,
        waves_per_eu=1,
    )
    return output


def qwen35_lm_head_argmax_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((1,), device=x.device, dtype=torch.int32)


direct_register_custom_op(
    op_name="qwen35_lm_head_argmax",
    op_func=qwen35_lm_head_argmax_impl,
    fake_impl=qwen35_lm_head_argmax_fake,
)
