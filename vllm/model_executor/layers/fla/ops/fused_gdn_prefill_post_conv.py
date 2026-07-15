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


@triton.jit
def _fused_post_conv_cumsum_4096_kernel(
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
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NT: tl.constexpr,
    NGT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    GATE_T: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Prepare the exact 4096-token shape and emit cumulative G directly.

    Q/K and V/beta retain the original 16-token work decomposition.  Gate
    programs own one complete 64-token GDN chunk, so the FP32 local cumsum is
    performed before the value ever reaches HBM.  Flattening the three task
    regions into one grid removes the separate cumsum launch without forcing
    the large Q/K/V tiles to use a 64-token register footprint.
    """
    pid = tl.program_id(0)
    qk_tasks: tl.constexpr = H * NT
    value_tasks: tl.constexpr = HV * NT

    if pid < qk_tasks:
        i_h = pid // NT
        i_tb = pid - i_h * NT
        q_t = i_tb * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_k = tl.arange(0, BK)
        mask_k = offs_k < K
        mask = mask_k[None, :]
        HK: tl.constexpr = H * K
        q_offsets = q_t[:, None] * stride_x_tok + i_h * K + offs_k[None, :]
        k_offsets = q_offsets + HK
        q_f32 = tl.load(
            mixed_qkv_ptr + q_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        k_f32 = tl.load(
            mixed_qkv_ptr + k_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        q_f32 *= (
            1.0 / tl.sqrt(tl.sum(q_f32 * q_f32, axis=1) + 1e-6)
        )[:, None]
        k_f32 *= (
            1.0 / tl.sqrt(tl.sum(k_f32 * k_f32, axis=1) + 1e-6)
        )[:, None]
        q_out = q_t[:, None] * stride_q_tok + i_h * K + offs_k[None, :]
        k_out = q_t[:, None] * stride_k_tok + i_h * K + offs_k[None, :]
        tl.store(q_ptr + q_out, q_f32, mask=mask)
        tl.store(k_ptr + k_out, k_f32, mask=mask)

    elif pid < qk_tasks + value_tasks:
        local = pid - qk_tasks
        i_hv = local // NT
        i_tb = local - i_hv * NT
        v_t = i_tb * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_v = tl.arange(0, BV)
        mask_v = offs_v < V
        mask = mask_v[None, :]
        V_OFFSET: tl.constexpr = 2 * H * K
        v_offsets = (
            v_t[:, None] * stride_x_tok
            + V_OFFSET
            + i_hv * V
            + offs_v[None, :]
        )
        v_vals = tl.load(mixed_qkv_ptr + v_offsets, mask=mask, other=0.0)
        v_out = v_t[:, None] * stride_v_tok + i_hv * V + offs_v[None, :]
        tl.store(v_ptr + v_out, v_vals, mask=mask)

        b_vals = tl.load(b_ptr + v_t * stride_b_tok + i_hv).to(tl.float32)
        beta_vals = tl.sigmoid(b_vals)
        tl.store(beta_ptr + v_t * HV + i_hv, beta_vals)

    else:
        local = pid - qk_tasks - value_tasks
        i_hv = local // NGT
        i_gt = local - i_hv * NGT
        g_t = i_gt * GATE_T + tl.arange(0, GATE_T)
        a_vals = tl.load(a_ptr + g_t * stride_a_tok + i_hv).to(tl.float32)
        A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
        x = a_vals + dt_bias_val
        softplus = tl.where(
            x > 0,
            x + tl.log(1.0 + tl.exp(-x)),
            tl.log(1.0 + tl.exp(x)),
        )
        softplus = tl.where(x <= 20.0, softplus, x)
        raw_g = -tl.exp(A_log_val) * softplus
        cumulative_g = tl.cumsum(raw_g, axis=0)
        tl.store(g_ptr + g_t * HV + i_hv, cumulative_g)


def can_use_exact_local_cumsum(
    conv_output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    apply_l2norm: bool,
    output_g_exp: bool,
    output_local_cumsum: bool,
) -> bool:
    """Whether the exact gfx936 4096-token local-cumsum route is valid."""
    from vllm.platforms.rocm import on_gfx936

    num_v_heads = A_log.shape[0] if A_log.ndim == 1 else -1
    expected_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    tensors = (conv_output, a, b, A_log, dt_bias)
    return (
        output_local_cumsum
        and on_gfx936()
        and conv_output.shape == (4096, expected_dim)
        and a.shape == b.shape == (4096, 48)
        and A_log.shape == dt_bias.shape == (48,)
        and num_k_heads == 16
        and num_v_heads == 48
        and head_k_dim == 128
        and head_v_dim == 128
        and conv_output.dtype == torch.bfloat16
        and a.dtype == b.dtype == torch.bfloat16
        and A_log.dtype in (torch.bfloat16, torch.float32)
        and dt_bias.dtype in (torch.bfloat16, torch.float32)
        and apply_l2norm
        and not output_g_exp
        and all(x.is_cuda for x in tensors)
        and all(x.device == conv_output.device for x in tensors[1:])
        and conv_output.stride(1) == 1
        and a.stride(1) == b.stride(1) == 1
        and A_log.is_contiguous()
        and dt_bias.is_contiguous()
    )


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
    output_local_cumsum: bool = False,
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
    use_exact_cumsum = can_use_exact_local_cumsum(
        conv_output=conv_output,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        apply_l2norm=apply_l2norm,
        output_g_exp=output_g_exp,
        output_local_cumsum=output_local_cumsum,
    )
    if use_exact_cumsum:
        block_t = 16
        gate_t = 64
        num_token_tiles = length // block_t
        num_gate_tiles = length // gate_t
        grid = (
            num_k_heads * num_token_tiles
            + num_v_heads * num_token_tiles
            + num_v_heads * num_gate_tiles,
        )
        _fused_post_conv_cumsum_4096_kernel[grid](
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
            H=num_k_heads,
            HV=num_v_heads,
            K=head_k_dim,
            V=head_v_dim,
            NT=num_token_tiles,
            NGT=num_gate_tiles,
            BLOCK_T=block_t,
            GATE_T=gate_t,
            BK=block_k,
            BV=block_v,
            num_warps=2,
            num_stages=1,
            waves_per_eu=1,
        )
        return q, k, v, g, beta

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
