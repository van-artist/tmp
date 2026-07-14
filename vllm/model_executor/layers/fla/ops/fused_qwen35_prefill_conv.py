# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact gfx936 Qwen3.5 prefill convolution/preparation fusion.

The guarded B=1, T=4096 path avoids materializing and rereading the packed
10240-channel causal-convolution output.  It produces Q/K/V, beta, the local
gate cumsum, and updates the width-4 convolution state in one launch.  Q/K
normalization retains an explicit BF16 store/load boundary inside that launch;
this reproduces the production convolution rounding bit for bit while removing
the standalone normalization launch.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


_T = 4096
_H = 16
_HV = 48
_K = 128
_V = 128
_WIDTH = 4
_QKV = 2 * _H * _K + _HV * _V


@triton.jit
def _conv_silu_bf16_tile(
    x_ptr,
    w_ptr,
    state_ptr,
    token_offsets,
    channel_offsets,
    cache_idx,
    has_initial,
    stride_x_tok: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_state_seq: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_tok: tl.constexpr,
    WIDTH: tl.constexpr,
):
    acc = tl.zeros(
        (token_offsets.shape[0], channel_offsets.shape[1]), dtype=tl.float32
    )
    for j in tl.static_range(0, WIDTH):
        source_tokens = token_offsets + j - (WIDTH - 1)
        from_x = tl.load(
            x_ptr + source_tokens * stride_x_tok + channel_offsets,
            mask=source_tokens >= 0,
            other=0.0,
        )
        # Only negative ``source_tokens`` consume state; for those lanes the
        # exact cache column is source_tokens + WIDTH - 1 in [0, WIDTH-2].
        # Keep masked non-negative lanes on a safe in-bounds address without
        # the HCU backend's unsupported scalar min/max lowering at BLOCK_T=16.
        state_columns = tl.where(
            source_tokens < 0, source_tokens + WIDTH - 1, 0
        )
        from_state = tl.load(
            state_ptr
            + cache_idx * stride_state_seq
            + channel_offsets * stride_state_dim
            + state_columns * stride_state_tok,
            mask=(source_tokens < 0) & has_initial,
            other=0.0,
        )
        values = tl.where(source_tokens < 0, from_state, from_x)
        weights = tl.load(
            w_ptr + channel_offsets * stride_w_dim + j * stride_w_width
        )
        acc += values * weights
    acc = acc / (1.0 + tl.exp(-acc))
    # Preserve the original convolution-output BF16 store/load boundary.
    return acc.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _conv_prep_cumsum_4096_kernel(
    x_ptr,
    w_ptr,
    state_ptr,
    state_indices_ptr,
    has_initial_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    stride_x_tok: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_state_seq: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_q_tok: tl.constexpr,
    stride_k_tok: tl.constexpr,
    stride_v_tok: tl.constexpr,
    QK_NT: tl.constexpr,
    NT: tl.constexpr,
    NGT: tl.constexpr,
    QK_BLOCK_T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    GATE_T: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    WIDTH: tl.constexpr,
    FUSE_QK_NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    qk_tasks: tl.constexpr = H * QK_NT
    value_tasks: tl.constexpr = HV * NT
    cache_idx = tl.load(state_indices_ptr).to(tl.int64)
    has_initial = tl.load(has_initial_ptr).to(tl.int1)

    if pid < qk_tasks:
        i_h = pid // QK_NT
        i_tb = pid - i_h * QK_NT
        tokens = i_tb * QK_BLOCK_T + tl.arange(0, QK_BLOCK_T)[:, None]
        features = tl.arange(0, K)[None, :]

        q_channels = i_h * K + features
        q_f32 = _conv_silu_bf16_tile(
            x_ptr,
            w_ptr,
            state_ptr,
            tokens,
            q_channels,
            cache_idx,
            has_initial,
            stride_x_tok,
            stride_w_dim,
            stride_w_width,
            stride_state_seq,
            stride_state_dim,
            stride_state_tok,
            WIDTH,
        )
        q_offsets = q_ptr + tokens * stride_q_tok + i_h * K + features
        if FUSE_QK_NORM:
            # Force the production BF16 materialization boundary.  A bare
            # bf16->f32 round trip can be folded by HCU when it remains in
            # registers; the same-program store/load is exact and still
            # removes the standalone normalization launch.
            tl.store(q_offsets, q_f32)
            tl.debug_barrier()
            q_f32 = tl.load(q_offsets).to(tl.float32)
            q_f32 *= (
                1.0 / tl.sqrt(tl.sum(q_f32 * q_f32, axis=1) + 1e-6)
            )[:, None]
        tl.store(q_offsets, q_f32)

        k_channels = H * K + i_h * K + features
        k_f32 = _conv_silu_bf16_tile(
            x_ptr,
            w_ptr,
            state_ptr,
            tokens,
            k_channels,
            cache_idx,
            has_initial,
            stride_x_tok,
            stride_w_dim,
            stride_w_width,
            stride_state_seq,
            stride_state_dim,
            stride_state_tok,
            WIDTH,
        )
        k_offsets = k_ptr + tokens * stride_k_tok + i_h * K + features
        if FUSE_QK_NORM:
            tl.store(k_offsets, k_f32)
            tl.debug_barrier()
            k_f32 = tl.load(k_offsets).to(tl.float32)
            k_f32 *= (
                1.0 / tl.sqrt(tl.sum(k_f32 * k_f32, axis=1) + 1e-6)
            )[:, None]
        tl.store(k_offsets, k_f32)

        # The first tile has already consumed the old state.  It is the sole
        # writer for both disjoint channel ranges, avoiding inter-program races.
        if i_tb == 0:
            state_col = tl.arange(0, 4)[:, None]
            state_mask = state_col < (WIDTH - 1)
            last_tokens = T - (WIDTH - 1) + state_col
            q_last = tl.load(
                x_ptr + last_tokens * stride_x_tok + q_channels,
                mask=state_mask,
            )
            k_last = tl.load(
                x_ptr + last_tokens * stride_x_tok + k_channels,
                mask=state_mask,
            )
            tl.store(
                state_ptr
                + cache_idx * stride_state_seq
                + q_channels * stride_state_dim
                + state_col * stride_state_tok,
                q_last,
                mask=state_mask,
            )
            tl.store(
                state_ptr
                + cache_idx * stride_state_seq
                + k_channels * stride_state_dim
                + state_col * stride_state_tok,
                k_last,
                mask=state_mask,
            )

    elif pid < qk_tasks + value_tasks:
        local = pid - qk_tasks
        i_hv = local // NT
        i_tb = local - i_hv * NT
        flat_tokens = i_tb * BLOCK_T + tl.arange(0, BLOCK_T)
        tokens = flat_tokens[:, None]
        features = tl.arange(0, V)[None, :]
        v_channels = 2 * H * K + i_hv * V + features
        v_f32 = _conv_silu_bf16_tile(
            x_ptr,
            w_ptr,
            state_ptr,
            tokens,
            v_channels,
            cache_idx,
            has_initial,
            stride_x_tok,
            stride_w_dim,
            stride_w_width,
            stride_state_seq,
            stride_state_dim,
            stride_state_tok,
            WIDTH,
        )
        tl.store(v_ptr + tokens * stride_v_tok + i_hv * V + features, v_f32)

        b_vals = tl.load(b_ptr + flat_tokens * stride_b_tok + i_hv).to(
            tl.float32
        )
        tl.store(beta_ptr + flat_tokens * HV + i_hv, tl.sigmoid(b_vals))

        if i_tb == 0:
            state_col = tl.arange(0, 4)[:, None]
            state_mask = state_col < (WIDTH - 1)
            last_tokens = T - (WIDTH - 1) + state_col
            v_last = tl.load(
                x_ptr + last_tokens * stride_x_tok + v_channels,
                mask=state_mask,
            )
            tl.store(
                state_ptr
                + cache_idx * stride_state_seq
                + v_channels * stride_state_dim
                + state_col * stride_state_tok,
                v_last,
                mask=state_mask,
            )

    else:
        local = pid - qk_tasks - value_tasks
        i_hv = local // NGT
        i_gt = local - i_hv * NGT
        gate_tokens = i_gt * GATE_T + tl.arange(0, GATE_T)
        a_vals = tl.load(a_ptr + gate_tokens * stride_a_tok + i_hv).to(
            tl.float32
        )
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
        tl.store(
            g_ptr + gate_tokens * HV + i_hv,
            tl.cumsum(raw_g, axis=0),
        )


@triton.jit
def _qk_norm_4096_inplace_kernel(
    q_ptr,
    k_ptr,
    stride_q_tok: tl.constexpr,
    stride_k_tok: tl.constexpr,
    NT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(0)
    i_h = pid // NT
    i_tb = pid - i_h * NT
    tokens = i_tb * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    features = tl.arange(0, K)[None, :]
    q_offsets = tokens * stride_q_tok + i_h * K + features
    k_offsets = tokens * stride_k_tok + i_h * K + features
    q_f32 = tl.load(q_ptr + q_offsets).to(tl.float32)
    k_f32 = tl.load(k_ptr + k_offsets).to(tl.float32)
    q_f32 *= (
        1.0 / tl.sqrt(tl.sum(q_f32 * q_f32, axis=1) + 1e-6)
    )[:, None]
    k_f32 *= (
        1.0 / tl.sqrt(tl.sum(k_f32 * k_f32, axis=1) + 1e-6)
    )[:, None]
    tl.store(q_ptr + q_offsets, q_f32)
    tl.store(k_ptr + k_offsets, k_f32)


def can_use_qwen35_prefill_conv_fusion(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state: torch.Tensor | None,
    state_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    activation: str,
) -> bool:
    """Return true only for the measured gfx936 B=1/T=4096 layout."""
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx936

    tensors = (
        mixed_qkv,
        conv_weight,
        conv_state,
        state_indices,
        has_initial_state,
        query_start_loc,
        a,
        b,
        A_log,
        dt_bias,
    )
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return False
    return (
        on_gfx936()
        and conv_bias is None
        and activation in ("silu", "swish")
        and all(t.is_cuda for t in tensors)
        and mixed_qkv.shape == (_T, _QKV)
        and mixed_qkv.dtype == torch.bfloat16
        # Qwen3.5 normally passes the qkv prefix of its packed 16480-wide
        # qkv/z/b/a projection.  The kernel addresses tokens through the
        # explicit stride, so both that production view and an isolated
        # contiguous fixture are exact.
        and mixed_qkv.stride(1) == 1
        and mixed_qkv.stride(0) in (_QKV, 16480)
        and conv_weight.shape == (_QKV, _WIDTH)
        and conv_weight.dtype == torch.bfloat16
        and conv_weight.is_contiguous()
        and conv_state.ndim == 3
        and conv_state.shape[1:] == (_QKV, _WIDTH - 1)
        and conv_state.dtype == torch.bfloat16
        and conv_state.stride(1) == 1
        and conv_state.stride(2) == _QKV
        and state_indices.shape == (1,)
        and state_indices.dtype == torch.int32
        and state_indices.is_contiguous()
        and has_initial_state.shape == (1,)
        and has_initial_state.dtype == torch.bool
        and has_initial_state.is_contiguous()
        and query_start_loc.shape == (2,)
        and query_start_loc.dtype == torch.int32
        and a.shape == (_T, _HV)
        and b.shape == (_T, _HV)
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and a.stride(1) == 1
        and b.stride(1) == 1
        and A_log.shape == (_HV,)
        and dt_bias.shape == (_HV,)
        and A_log.dtype in (torch.bfloat16, torch.float32)
        and dt_bias.dtype in (torch.bfloat16, torch.float32)
        and A_log.is_contiguous()
        and dt_bias.is_contiguous()
        and all(t.device == mixed_qkv.device for t in tensors)
    )


def fused_qwen35_prefill_conv_prep(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    fuse_qk_norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the already-guarded exact B=1/T=4096 convolution fusion.

    ``fuse_qk_norm=False`` retains the prior two-launch implementation for
    target-device A/B and bitwise validation.  The production default keeps
    the convolution BF16 rounding boundary inside the same launch, then
    normalizes Q/K without a separate kernel launch.
    """
    q = torch.empty((_T, _H, _K), device=mixed_qkv.device, dtype=mixed_qkv.dtype)
    k = torch.empty_like(q)
    v = torch.empty(
        (_T, _HV, _V), device=mixed_qkv.device, dtype=mixed_qkv.dtype
    )
    g = torch.empty((_T, _HV), device=mixed_qkv.device, dtype=torch.float32)
    beta = torch.empty_like(g)

    # The standalone exact Q/K norm uses 16 tokens per program.  Match that
    # geometry in the fused specialization so Triton's warp reduction order
    # remains bitwise identical; retain the measured 8-token convolution
    # geometry for the reference/fallback path and all value heads.
    qk_block_t = 16 if fuse_qk_norm else 8
    qk_nt = _T // qk_block_t
    block_t = 8
    nt = _T // block_t
    gate_t = 64
    ngt = _T // gate_t
    grid = (_H * qk_nt + _HV * nt + _HV * ngt,)
    _conv_prep_cumsum_4096_kernel[grid](
        mixed_qkv,
        conv_weight,
        conv_state,
        state_indices,
        has_initial_state,
        a,
        b,
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        mixed_qkv.stride(0),
        conv_weight.stride(0),
        conv_weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        a.stride(0),
        b.stride(0),
        q.stride(0),
        k.stride(0),
        v.stride(0),
        QK_NT=qk_nt,
        NT=nt,
        NGT=ngt,
        QK_BLOCK_T=qk_block_t,
        BLOCK_T=block_t,
        GATE_T=gate_t,
        T=_T,
        H=_H,
        HV=_HV,
        K=_K,
        V=_V,
        WIDTH=_WIDTH,
        FUSE_QK_NORM=fuse_qk_norm,
        num_warps=2,
        num_stages=1,
        waves_per_eu=1,
    )

    if not fuse_qk_norm:
        norm_block_t = 16
        norm_nt = _T // norm_block_t
        _qk_norm_4096_inplace_kernel[(_H * norm_nt,)](
            q,
            k,
            q.stride(0),
            k.stride(0),
            NT=norm_nt,
            BLOCK_T=norm_block_t,
            H=_H,
            K=_K,
            num_warps=2,
            num_stages=1,
            waves_per_eu=1,
        )
    return q, k, v, g, beta
