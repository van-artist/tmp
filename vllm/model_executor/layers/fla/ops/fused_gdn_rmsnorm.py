# SPDX-License-Identifier: Apache-2.0
"""Exact Qwen3.5 GDN RMSNorm+gate for a strided six-way projection view."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rmsnorm_gated_strided_z_kernel(
    x_ptr,
    z_ptr,
    weight_ptr,
    stride_x_token,
    stride_x_head,
    stride_z_token,
    stride_z_head,
    num_tokens,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // num_heads
    head = row - token * num_heads
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_size

    x_offsets = token * stride_x_token + head * stride_x_head + offs_d
    z_offsets = token * stride_z_token + head * stride_z_head + offs_d
    x = tl.load(x_ptr + x_offsets, mask=mask_d, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + z_offsets, mask=mask_d, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs_d, mask=mask_d, other=0.0).to(
        tl.float32
    )

    variance = tl.sum(tl.where(mask_d, x * x, 0.0), axis=0) / head_size
    normalized = x * tl.rsqrt(variance + eps) * weight
    output = normalized * z * tl.sigmoid(z)
    tl.store(x_ptr + x_offsets, output, mask=mask_d)


def rmsnorm_gated_strided_z_inplace(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply ``RMSNorm(x) * SiLU(z)`` in-place without flattening ``z``.

    The Qwen3.5 six-way projection leaves z as a last-dimension-contiguous
    ``[token, value_head, dim]`` view whose token stride includes the packed
    b/a tail. Flattening that view materializes a 48 MiB copy at T=4096.
    """
    if x.ndim != 3 or z.ndim != 3 or x.shape != z.shape:
        raise ValueError("x and z must have the same [token, head, dim] shape")
    if x.stride(-1) != 1 or z.stride(-1) != 1:
        raise ValueError("x and z must be contiguous in their last dimension")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous for the in-place specialization")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must match the GDN head dimension")
    if x.numel() == 0:
        return x

    num_tokens, num_heads, head_size = x.shape
    block_d = triton.next_power_of_2(head_size)
    _rmsnorm_gated_strided_z_kernel[(num_tokens * num_heads,)](
        x_ptr=x,
        z_ptr=z,
        weight_ptr=weight,
        stride_x_token=x.stride(0),
        stride_x_head=x.stride(1),
        stride_z_token=z.stride(0),
        stride_z_head=z.stride(1),
        num_tokens=num_tokens,
        num_heads=num_heads,
        head_size=head_size,
        eps=eps,
        BLOCK_D=block_d,
        num_warps=min(max(block_d // 64, 1), 8),
        num_stages=1,
    )
    return x
