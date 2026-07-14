# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused split, Q/K RMSNorm and partial NeoX RoPE.

Backported from upstream vLLM for the Qwen3.5 full-attention layers.  It is
mathematically equivalent to the original sequence and keeps the BF16
round-trip before RoPE so the storage semantics remain the same.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_qk_rmsnorm_rope_gate_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_gate_stride_t,
    k_stride_t,
    q_out_stride_t,
    k_out_stride_t,
    cache_stride_p,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)

    if is_k:
        in_base = k_ptr + token * k_stride_t + local_head * head_dim
        weight_ptr = k_weight_ptr
        out_base = k_out_ptr + token * k_out_stride_t + local_head * head_dim
    else:
        in_base = (
            q_gate_ptr
            + token * q_gate_stride_t
            + local_head * 2 * head_dim
        )
        weight_ptr = q_weight_ptr
        out_base = q_out_ptr + token * q_out_stride_t + local_head * head_dim

    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < head_dim
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(
        tl.float32
    )
    variance = tl.sum(x * x, axis=0) / head_dim
    inv_rms = tl.rsqrt(variance + eps)
    # Gemma RMSNorm stores the offset from one.  Add one after the raw weight
    # load inside this existing kernel instead of materializing two temporary
    # FP32 tensors (and their conversion/add launches) in every attention
    # layer on every scheduler step.
    weight = tl.load(
        weight_ptr + head_offs, mask=head_mask, other=0.0
    ).to(tl.float32) + 1.0

    # Match the unfused GemmaRMSNorm output being stored as BF16 before RoPE.
    x_norm = (x * inv_rms * weight).to(INPUT_DTYPE).to(tl.float32)
    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= rotary_dim)
        tl.store(out_base + head_offs, x_norm, mask=pass_mask)

    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < half_rotary
    x_rot1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    x_rot2 = tl.load(
        in_base + half_rotary + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    w_rot1 = tl.load(
        weight_ptr + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32) + 1.0
    w_rot2 = tl.load(
        weight_ptr + half_rotary + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32) + 1.0
    x_rot1 = (x_rot1 * inv_rms * w_rot1).to(INPUT_DTYPE).to(tl.float32)
    x_rot2 = (x_rot2 * inv_rms * w_rot2).to(INPUT_DTYPE).to(tl.float32)

    pos = tl.load(positions_ptr + token).to(tl.int64)
    cache_offset = pos * cache_stride_p
    cos = tl.load(
        cos_sin_cache_ptr + cache_offset + rot_offs,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_offset + half_rotary + rot_offs,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)
    out1 = x_rot1 * cos - x_rot2 * sin
    out2 = x_rot2 * cos + x_rot1 * sin
    tl.store(out_base + rot_offs, out1, mask=rot_mask)
    tl.store(out_base + half_rotary + rot_offs, out2, mask=rot_mask)


def fused_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Q/K and a zero-copy strided gate view.

    Qwen stores each query head as ``[q, gate]``.  The packed projection stays
    live until gating finishes, so copying all gate values into another dense
    tensor only adds an HBM round trip.  Returning the per-head view preserves
    the exact BF16 values and lets the caller fuse sigmoid/multiply while
    reading the original projection directly.
    """
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            "rotary_dim must be a positive even integer <= head_dim, "
            f"got rotary_dim={rotary_dim}, head_dim={head_dim}"
        )
    num_tokens = q_gate.shape[0]
    q_out = torch.empty(
        (num_tokens, num_q_heads * head_dim),
        dtype=q_gate.dtype,
        device=q_gate.device,
    )
    k_out = torch.empty(
        (num_tokens, num_kv_heads * head_dim),
        dtype=k.dtype,
        device=k.device,
    )
    gate_view = q_gate.view(num_tokens, num_q_heads, 2 * head_dim)[
        ..., head_dim:
    ]
    if num_tokens == 0:
        return q_out, k_out, gate_view

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rot_half_block = triton.next_power_of_2(half_rotary)
    num_warps = max(1, head_block // 64)
    grid = (num_tokens, num_q_heads + num_kv_heads)
    _fused_qk_rmsnorm_rope_gate_kernel[grid](
        q_gate,
        k,
        q_out,
        k_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        cos_sin_cache.stride(0),
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        half_rotary,
        eps,
        INPUT_DTYPE=(
            tl.bfloat16 if q_gate.dtype == torch.bfloat16 else tl.float16
        ),
        HEAD_BLOCK=head_block,
        ROT_HALF_BLOCK=rot_half_block,
        HAS_PASS=rotary_dim < head_dim,
        num_warps=num_warps,
        num_stages=2,
    )
    return q_out, k_out, gate_view
