# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""Exact Qwen3.5-27B prefill fusion for GDN solve + W/U recomputation.

This module is deliberately shape-specialized.  The public GDN API must keep
using ``solve_tril`` + ``recompute_w_u_fwd`` unless ``can_use_fused`` returns
true.  The kernel preserves the materialized BF16 inverse boundary via an
explicit RTNE cast, which makes W/U, the final output, and recurrent state
bitwise identical to the original path on gfx936 when launched with w2/s1.
"""

from __future__ import annotations

# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from .solve_tril import FLA_TRIL_PRECISION


@triton.jit
def _emit_w_u(
    k,
    v,
    beta,
    g,
    w,
    u,
    i_t,
    i_h,
    a11,
    a21,
    a22,
    a31,
    a32,
    a33,
    a41,
    a42,
    a43,
    a44,
    T: tl.constexpr,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    oi = tl.arange(0, 16)
    b1 = tl.load(beta + (i_t * BT + oi) * H + i_h)
    b2 = tl.load(beta + (i_t * BT + 16 + oi) * H + i_h)
    b3 = tl.load(beta + (i_t * BT + 32 + oi) * H + i_h)
    b4 = tl.load(beta + (i_t * BT + 48 + oi) * H + i_h)
    eg1 = tl.exp(tl.load(g + (i_t * BT + oi) * H + i_h))
    eg2 = tl.exp(tl.load(g + (i_t * BT + 16 + oi) * H + i_h))
    eg3 = tl.exp(tl.load(g + (i_t * BT + 32 + oi) * H + i_h))
    eg4 = tl.exp(tl.load(g + (i_t * BT + 48 + oi) * H + i_h))

    for iv in range(tl.cdiv(V, BV)):
        v0 = v + i_h * V
        pv1 = tl.make_block_ptr(v0, (T, V), (H * V, 1), (i_t * BT, iv * BV), (16, BV), (1, 0))
        pv2 = tl.make_block_ptr(v0, (T, V), (H * V, 1), (i_t * BT + 16, iv * BV), (16, BV), (1, 0))
        pv3 = tl.make_block_ptr(v0, (T, V), (H * V, 1), (i_t * BT + 32, iv * BV), (16, BV), (1, 0))
        pv4 = tl.make_block_ptr(v0, (T, V), (H * V, 1), (i_t * BT + 48, iv * BV), (16, BV), (1, 0))
        x1 = (tl.load(pv1) * b1[:, None]).to(tl.bfloat16)
        x2 = (tl.load(pv2) * b2[:, None]).to(tl.bfloat16)
        x3 = (tl.load(pv3) * b3[:, None]).to(tl.bfloat16)
        x4 = (tl.load(pv4) * b4[:, None]).to(tl.bfloat16)
        y1 = tl.dot(a11, x1, allow_tf32=False)
        y2 = tl.dot(a21, x1, allow_tf32=False) + tl.dot(a22, x2, allow_tf32=False)
        y3 = tl.dot(a31, x1, allow_tf32=False) + tl.dot(a32, x2, allow_tf32=False)
        y3 += tl.dot(a33, x3, allow_tf32=False)
        y4 = tl.dot(a41, x1, allow_tf32=False) + tl.dot(a42, x2, allow_tf32=False)
        y4 += tl.dot(a43, x3, allow_tf32=False)
        y4 += tl.dot(a44, x4, allow_tf32=False)
        u0 = u + i_h * V
        pu1 = tl.make_block_ptr(u0, (T, V), (H * V, 1), (i_t * BT, iv * BV), (16, BV), (1, 0))
        pu2 = tl.make_block_ptr(u0, (T, V), (H * V, 1), (i_t * BT + 16, iv * BV), (16, BV), (1, 0))
        pu3 = tl.make_block_ptr(u0, (T, V), (H * V, 1), (i_t * BT + 32, iv * BV), (16, BV), (1, 0))
        pu4 = tl.make_block_ptr(u0, (T, V), (H * V, 1), (i_t * BT + 48, iv * BV), (16, BV), (1, 0))
        tl.store(pu1, y1.to(tl.bfloat16))
        tl.store(pu2, y2.to(tl.bfloat16))
        tl.store(pu3, y3.to(tl.bfloat16))
        tl.store(pu4, y4.to(tl.bfloat16))

    for ik in range(tl.cdiv(K, BK)):
        ihg = i_h // (H // Hg)
        k0 = k + ihg * K
        pk1 = tl.make_block_ptr(k0, (T, K), (Hg * K, 1), (i_t * BT, ik * BK), (16, BK), (1, 0))
        pk2 = tl.make_block_ptr(k0, (T, K), (Hg * K, 1), (i_t * BT + 16, ik * BK), (16, BK), (1, 0))
        pk3 = tl.make_block_ptr(k0, (T, K), (Hg * K, 1), (i_t * BT + 32, ik * BK), (16, BK), (1, 0))
        pk4 = tl.make_block_ptr(k0, (T, K), (Hg * K, 1), (i_t * BT + 48, ik * BK), (16, BK), (1, 0))
        x1 = (tl.load(pk1) * b1[:, None] * eg1[:, None]).to(tl.bfloat16)
        x2 = (tl.load(pk2) * b2[:, None] * eg2[:, None]).to(tl.bfloat16)
        x3 = (tl.load(pk3) * b3[:, None] * eg3[:, None]).to(tl.bfloat16)
        x4 = (tl.load(pk4) * b4[:, None] * eg4[:, None]).to(tl.bfloat16)
        y1 = tl.dot(a11, x1)
        y2 = tl.dot(a21, x1) + tl.dot(a22, x2)
        y3 = tl.dot(a31, x1) + tl.dot(a32, x2)
        y3 += tl.dot(a33, x3)
        y4 = tl.dot(a41, x1) + tl.dot(a42, x2)
        y4 += tl.dot(a43, x3)
        y4 += tl.dot(a44, x4)
        w0 = w + i_h * K
        pw1 = tl.make_block_ptr(w0, (T, K), (H * K, 1), (i_t * BT, ik * BK), (16, BK), (1, 0))
        pw2 = tl.make_block_ptr(w0, (T, K), (H * K, 1), (i_t * BT + 16, ik * BK), (16, BK), (1, 0))
        pw3 = tl.make_block_ptr(w0, (T, K), (H * K, 1), (i_t * BT + 32, ik * BK), (16, BK), (1, 0))
        pw4 = tl.make_block_ptr(w0, (T, K), (H * K, 1), (i_t * BT + 48, ik * BK), (16, BK), (1, 0))
        tl.store(pw1, y1.to(tl.bfloat16))
        tl.store(pw2, y2.to(tl.bfloat16))
        tl.store(pw3, y3.to(tl.bfloat16))
        tl.store(pw4, y4.to(tl.bfloat16))


@triton.jit
def _solve_recompute_kernel(
    A,
    k,
    v,
    beta,
    g,
    w,
    u,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    oi = tl.arange(0, 16)
    lower = oi[:, None] > oi[None, :]
    ident = oi[:, None] == oi[None, :]
    A0 = A + i_h * BT
    p11 = tl.make_block_ptr(A0, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p22 = tl.make_block_ptr(A0, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p33 = tl.make_block_ptr(A0, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p44 = tl.make_block_ptr(A0, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    ai11 = -tl.where(lower, tl.load(p11).to(tl.float32), 0.0)
    ai22 = -tl.where(lower, tl.load(p22).to(tl.float32), 0.0)
    ai33 = -tl.where(lower, tl.load(p33).to(tl.float32), 0.0)
    ai44 = -tl.where(lower, tl.load(p44).to(tl.float32), 0.0)
    for i in range(2, 16):
        x = -tl.load(A0 + (i_t * BT + i) * H * BT + oi)
        x += tl.sum(x[:, None] * ai11, 0)
        ai11 = tl.where((oi == i)[:, None], x, ai11)
    for i in range(18, 32):
        x = -tl.load(A0 + (i_t * BT + i) * H * BT + oi + 16)
        x += tl.sum(x[:, None] * ai22, 0)
        ai22 = tl.where((oi == i - 16)[:, None], x, ai22)
    for i in range(34, 48):
        x = -tl.load(A0 + (i_t * BT + i) * H * BT + oi + 32)
        x += tl.sum(x[:, None] * ai33, 0)
        ai33 = tl.where((oi == i - 32)[:, None], x, ai33)
    for i in range(50, 64):
        x = -tl.load(A0 + (i_t * BT + i) * H * BT + oi + 48)
        x += tl.sum(x[:, None] * ai44, 0)
        ai44 = tl.where((oi == i - 48)[:, None], x, ai44)
    ai11 += ident
    ai22 += ident
    ai33 += ident
    ai44 += ident

    p21 = tl.make_block_ptr(A0, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p31 = tl.make_block_ptr(A0, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p32 = tl.make_block_ptr(A0, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p41 = tl.make_block_ptr(A0, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p42 = tl.make_block_ptr(A0, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p43 = tl.make_block_ptr(A0, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    a21 = tl.load(p21).to(tl.float32)
    a31 = tl.load(p31).to(tl.float32)
    a32 = tl.load(p32).to(tl.float32)
    a41 = tl.load(p41).to(tl.float32)
    a42 = tl.load(p42).to(tl.float32)
    a43 = tl.load(p43).to(tl.float32)
    ai21 = -tl.dot(tl.dot(ai22, a21, input_precision=DOT_PRECISION), ai11, input_precision=DOT_PRECISION)
    ai32 = -tl.dot(tl.dot(ai33, a32, input_precision=DOT_PRECISION), ai22, input_precision=DOT_PRECISION)
    ai43 = -tl.dot(tl.dot(ai44, a43, input_precision=DOT_PRECISION), ai33, input_precision=DOT_PRECISION)
    ai31 = -tl.dot(ai33, tl.dot(a31, ai11, input_precision=DOT_PRECISION) + tl.dot(a32, ai21, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
    ai42 = -tl.dot(ai44, tl.dot(a42, ai22, input_precision=DOT_PRECISION) + tl.dot(a43, ai32, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)
    ai41 = -tl.dot(ai44, tl.dot(a41, ai11, input_precision=DOT_PRECISION) + tl.dot(a42, ai21, input_precision=DOT_PRECISION) + tl.dot(a43, ai31, input_precision=DOT_PRECISION), input_precision=DOT_PRECISION)

    _emit_w_u(
        k, v, beta, g, w, u, i_t, i_h,
        ai11.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        ai21.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        ai22.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        ai31.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        ai32.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        ai33.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        ai41.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        ai42.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        ai43.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        ai44.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        T=T, H=H, Hg=Hg, K=K, V=V, BT=BT, BK=BK, BV=BV,
    )


def can_use_fused(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
) -> bool:
    from vllm.platforms.rocm import on_gfx936

    return (
        on_gfx936()
        and k.shape == (1, 4096, 16, 128)
        and v.shape == (1, 4096, 48, 128)
        and beta.shape == (1, 4096, 48)
        and g.shape == (1, 4096, 48)
        and A.shape == (1, 4096, 48, 64)
        and k.dtype == v.dtype == torch.bfloat16
        and beta.dtype == g.dtype == torch.float32
        and A.dtype == torch.float32
        and all(x.is_cuda for x in (k, v, beta, g, A))
        and all(x.device == k.device for x in (v, beta, g, A))
        and all(x.is_contiguous() for x in (k, v, beta, g, A))
    )


def solve_recompute_fused(
    A: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not can_use_fused(k, v, beta, g, A):
        raise ValueError("fused solve/recompute received an unsupported shape or dtype")
    w = k.new_empty(1, 4096, 48, 128)
    u = torch.empty_like(v)
    _solve_recompute_kernel[(64, 48)](
        A, k, v, beta, g, w, u,
        H=48, Hg=16, K=128, V=128, T=4096, BT=64, BK=64, BV=64,
        DOT_PRECISION=FLA_TRIL_PRECISION,
        num_warps=2,
        num_stages=1,
    )
    return w, u
