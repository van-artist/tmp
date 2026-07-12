# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused post-conv preparation for the Qwen3.5 GDN prefill path.

This is a targeted backport of the upstream vLLM fusion.  It replaces the
split/rearrange/contiguous, two L2 normalizations, and gate preparation with
one Triton launch.  The recurrent GDN calculation itself is unchanged.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_post_conv_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    stride_x_tok,
    stride_a_tok,
    stride_b_tok,
    stride_q_tok,
    stride_k_tok,
    stride_v_tok,
    L,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    APPLY_L2NORM: tl.constexpr,
    L2NORM_EPS: tl.constexpr,
    OUTPUT_G_EXP: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Prepare Q/K/V and the two gates directly in their target layouts."""
    i_tb = tl.program_id(0)
    i_head = tl.program_id(1)
    HK: tl.constexpr = H * K
    offs_t = i_tb * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < L

    if i_head < H:
        i_h = i_head
        offs_k = tl.arange(0, BK)
        mask_k = offs_k < K
        mask_2d = mask_t[:, None] & mask_k[None, :]

        q_offsets = (
            offs_t[:, None] * stride_x_tok + i_h * K + offs_k[None, :]
        )
        q_f32 = tl.load(
            mixed_qkv_ptr + q_offsets, mask=mask_2d, other=0
        ).to(tl.float32)

        k_offsets = (
            offs_t[:, None] * stride_x_tok + HK + i_h * K + offs_k[None, :]
        )
        k_f32 = tl.load(
            mixed_qkv_ptr + k_offsets, mask=mask_2d, other=0
        ).to(tl.float32)

        if APPLY_L2NORM:
            q_sq_sum = tl.sum(q_f32 * q_f32, axis=1)
            q_f32 = q_f32 * (1.0 / tl.sqrt(q_sq_sum + L2NORM_EPS))[:, None]
            k_sq_sum = tl.sum(k_f32 * k_f32, axis=1)
            k_f32 = k_f32 * (1.0 / tl.sqrt(k_sq_sum + L2NORM_EPS))[:, None]

        q_out = offs_t[:, None] * stride_q_tok + i_h * K + offs_k[None, :]
        tl.store(
            q_ptr + q_out,
            q_f32.to(q_ptr.dtype.element_ty),
            mask=mask_2d,
        )
        k_out = offs_t[:, None] * stride_k_tok + i_h * K + offs_k[None, :]
        tl.store(
            k_ptr + k_out,
            k_f32.to(k_ptr.dtype.element_ty),
            mask=mask_2d,
        )
    else:
        i_hv = i_head - H
        offs_v = tl.arange(0, BV)
        mask_v = offs_v < V
        mask_2d = mask_t[:, None] & mask_v[None, :]
        V_OFFSET: tl.constexpr = 2 * H * K
        v_offsets = (
            offs_t[:, None] * stride_x_tok
            + V_OFFSET
            + i_hv * V
            + offs_v[None, :]
        )
        v_vals = tl.load(mixed_qkv_ptr + v_offsets, mask=mask_2d, other=0)
        v_out = offs_t[:, None] * stride_v_tok + i_hv * V + offs_v[None, :]
        tl.store(v_ptr + v_out, v_vals, mask=mask_2d)

        A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
        a_offsets = offs_t * stride_a_tok + i_hv
        b_offsets = offs_t * stride_b_tok + i_hv
        a_vals = tl.load(a_ptr + a_offsets, mask=mask_t, other=0).to(
            tl.float32
        )
        b_vals = tl.load(b_ptr + b_offsets, mask=mask_t, other=0).to(
            tl.float32
        )
        x = a_vals + dt_bias_val
        softplus = tl.where(
            x > 0,
            x + tl.log(1.0 + tl.exp(-x)),
            tl.log(1.0 + tl.exp(x)),
        )
        softplus = tl.where(x <= SOFTPLUS_THRESHOLD, softplus, x)
        g_vals = -tl.exp(A_log_val) * softplus
        if OUTPUT_G_EXP:
            g_vals = tl.exp(g_vals)
        beta_vals = tl.sigmoid(b_vals)
        gb_offsets = offs_t * HV + i_hv
        tl.store(g_ptr + gb_offsets, g_vals, mask=mask_t)
        tl.store(beta_ptr + gb_offsets, beta_vals, mask=mask_t)


def fused_post_conv_prep(
    conv_output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fuse Q/K/V layout conversion, Q/K L2 norm and GDN gates."""
    length = conv_output.shape[0]
    qkv_dim = conv_output.shape[1]
    num_v_heads = A_log.shape[0]
    expected_dim = (
        2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    )
    assert qkv_dim == expected_dim, (
        f"qkv_dim={qkv_dim} != 2*H*K + HV*V = {expected_dim}"
    )

    dtype = conv_output.dtype
    device = conv_output.device
    q = torch.empty(
        length, num_k_heads, head_k_dim, dtype=dtype, device=device
    )
    k = torch.empty_like(q)
    v = torch.empty(
        length, num_v_heads, head_v_dim, dtype=dtype, device=device
    )
    g = torch.empty(
        length, num_v_heads, dtype=torch.float32, device=device
    )
    beta = torch.empty_like(g)
    if length == 0:
        return q, k, v, g, beta

    block_k = triton.next_power_of_2(head_k_dim)
    block_v = triton.next_power_of_2(head_v_dim)
    block_t = 16
    grid = (triton.cdiv(length, block_t), num_k_heads + num_v_heads)
    _fused_post_conv_kernel[grid](
        mixed_qkv_ptr=conv_output,
        a_ptr=a,
        b_ptr=b,
        A_log_ptr=A_log,
        dt_bias_ptr=dt_bias,
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        g_ptr=g,
        beta_ptr=beta,
        stride_x_tok=conv_output.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_q_tok=q.stride(0),
        stride_k_tok=k.stride(0),
        stride_v_tok=v.stride(0),
        L=length,
        H=num_k_heads,
        HV=num_v_heads,
        K=head_k_dim,
        V=head_v_dim,
        APPLY_L2NORM=apply_l2norm,
        L2NORM_EPS=1e-6,
        OUTPUT_G_EXP=output_g_exp,
        SOFTPLUS_THRESHOLD=20.0,
        BLOCK_T=block_t,
        BK=block_k,
        BV=block_v,
        num_warps=4,
        num_stages=2,
    )
    return q, k, v, g, beta
