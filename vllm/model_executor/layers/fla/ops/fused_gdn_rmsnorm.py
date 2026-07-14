# SPDX-License-Identifier: Apache-2.0
"""Exact Qwen3.5 GDN RMSNorm+gate for a strided six-way projection view."""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


# Resolve the hardware choice once at import time.  This deliberately avoids
# consulting the dynamic token extent from Python, so one compiled graph can
# serve decode and every prefill bucket.
if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx936

    _USE_GFX936_QWEN35_HEAD_TILE = on_gfx936()
else:
    _USE_GFX936_QWEN35_HEAD_TILE = False


@triton.jit
def _rmsnorm_gated_strided_z_kernel(
    x_ptr,
    z_ptr,
    weight_ptr,
    stride_x_token,
    stride_x_head,
    stride_z_token,
    stride_z_head,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
):
    # Keep the scheduler's symbolic token extent as its own launch dimension.
    # Flattening token*head into a 1-D grid turns the profiled T<=4096 range
    # into 4096*48=196608 and is rejected by vLLM's dynamic-shape guard during
    # EngineCore initialization.
    token = tl.program_id(0)
    head = tl.program_id(1)
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


@triton.jit
def _rmsnorm_gated_strided_z_headtile_kernel(
    x_ptr,
    z_ptr,
    weight_ptr,
    stride_x_token,
    stride_x_head,
    stride_z_token,
    stride_z_head,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Process four independent value heads in one program.

    Keeping token as grid dimension zero makes the launch compatible with
    vLLM's symbolic T range.  Each head retains exactly the original FP32
    reduction order along D.
    """
    token = tl.program_id(0)
    head0 = tl.program_id(1) * BLOCK_H
    offs_h = head0 + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_h[:, None] < num_heads) & (offs_d[None, :] < head_size)

    x_offsets = (
        token * stride_x_token
        + offs_h[:, None] * stride_x_head
        + offs_d[None, :]
    )
    z_offsets = (
        token * stride_z_token
        + offs_h[:, None] * stride_z_head
        + offs_d[None, :]
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + z_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(
        weight_ptr + offs_d[None, :],
        mask=offs_d[None, :] < head_size,
        other=0.0,
    ).to(tl.float32)

    variance = tl.sum(tl.where(mask, x * x, 0.0), axis=1) / head_size
    normalized = x * tl.rsqrt(variance[:, None] + eps) * weight
    output = normalized * z * tl.sigmoid(z)
    tl.store(x_ptr + x_offsets, output, mask=mask)


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
    use_qwen35_head_tile = (
        _USE_GFX936_QWEN35_HEAD_TILE
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and z.dtype == x.dtype
        and weight.dtype == x.dtype
        and z.device == x.device
        and weight.device == x.device
        and num_heads == 48
        and head_size == 128
        and x.stride(0) == 6144
        and x.stride(1) == 128
        and z.stride(0) == 16480
        and z.stride(1) == 128
        and weight.is_contiguous()
    )
    if use_qwen35_head_tile:
        # T remains grid dimension zero for decode and all prefill buckets.
        block_h = 4
        _rmsnorm_gated_strided_z_headtile_kernel[
            (num_tokens, triton.cdiv(num_heads, block_h))
        ](
            x_ptr=x,
            z_ptr=z,
            weight_ptr=weight,
            stride_x_token=x.stride(0),
            stride_x_head=x.stride(1),
            stride_z_token=z.stride(0),
            stride_z_head=z.stride(1),
            num_heads=num_heads,
            head_size=head_size,
            eps=eps,
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            num_warps=8,
            num_stages=1,
            waves_per_eu=1,
        )
        return x

    _rmsnorm_gated_strided_z_kernel[(num_tokens, num_heads)](
        x_ptr=x,
        z_ptr=z,
        weight_ptr=weight,
        stride_x_token=x.stride(0),
        stride_x_head=x.stride(1),
        stride_z_token=z.stride(0),
        stride_z_head=z.stride(1),
        num_heads=num_heads,
        head_size=head_size,
        eps=eps,
        BLOCK_D=block_d,
        num_warps=min(max(block_d // 64, 1), 8),
        num_stages=1,
    )
    return x
