# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# The MFMA tiling and launch configuration are adapted from AMD AITER's
# gemm_a16w16 kernel.  The dual accumulation and SwiGLU epilogue are local to
# this exact Qwen3.5 specialization, which keeps the original BF16 weights and
# reproduces vLLM SiluAndMul's BF16 rounding order.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm import envs
from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

_QWEN35_HIDDEN_SIZE = 5120
_QWEN35_INTERMEDIATE_SIZE = 17408
_QWEN35_GATE_UP_SIZE = 2 * _QWEN35_INTERMEDIATE_SIZE


@triton.jit
def _qwen35_bf16_gateup_silu_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    # M is exactly one.  Replicating its row to a 16-row MFMA tile matches the
    # production A16W16 GEMV reduction while only row zero is written.
    x_ptrs = x_ptr + offs_m[:, None] * 0 + offs_k[None, :]
    gate_ptrs = weight_ptr + offs_d[None, :] * K + offs_k[:, None]
    up_ptrs = weight_ptr + (offs_d[None, :] + D) * K + offs_k[:, None]
    acc_gate = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs)
        gate_w = tl.load(gate_ptrs, mask=offs_d[None, :] < D, other=0.0)
        up_w = tl.load(up_ptrs, mask=offs_d[None, :] < D, other=0.0)
        acc_gate += tl.dot(x, gate_w, input_precision="ieee")
        acc_up += tl.dot(x, up_w, input_precision="ieee")
        x_ptrs += BLOCK_K
        gate_ptrs += BLOCK_K
        up_ptrs += BLOCK_K

    # The unfused path writes both GEMM halves as BF16, computes SiLU in FP32,
    # rounds the activated gate to BF16, then performs the BF16 multiply.  Keep
    # both explicit round trips so this fusion is bitwise-equivalent.
    gate = acc_gate.to(tl.bfloat16).to(tl.float32)
    up = acc_up.to(tl.bfloat16).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    result = activated * up

    out_ptrs = output_ptr + offs_m[:, None] * D + offs_d[None, :]
    tl.store(
        out_ptrs,
        result,
        mask=(offs_m[:, None] == 0) & (offs_d[None, :] < D),
    )


def qwen35_bf16_gateup_silu_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    num_tokens = x.numel() // x.shape[-1]
    use_fused = (
        envs.VLLM_ROCM_QWEN35_FUSED_GATEUP_SILU
        and current_platform.is_rocm()
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == x.dtype
        and weight.device == x.device
        and x.shape[-1] == _QWEN35_HIDDEN_SIZE
        and weight.shape == (_QWEN35_GATE_UP_SIZE, _QWEN35_HIDDEN_SIZE)
        and num_tokens == 1
        and x.is_contiguous()
        and weight.is_contiguous()
    )
    if use_fused:
        from vllm.platforms.rocm import on_gfx936

        use_fused = on_gfx936()

    if use_fused:
        x_2d = x.reshape(1, _QWEN35_HIDDEN_SIZE)
        output = x.new_empty((1, _QWEN35_INTERMEDIATE_SIZE))
        _qwen35_bf16_gateup_silu_kernel[
            (triton.cdiv(_QWEN35_INTERMEDIATE_SIZE, 16),)
        ](
            x_2d,
            weight,
            output,
            K=_QWEN35_HIDDEN_SIZE,
            D=_QWEN35_INTERMEDIATE_SIZE,
            BLOCK_M=16,
            BLOCK_D=16,
            BLOCK_K=512,
            num_warps=4,
            num_stages=1,
            waves_per_eu=1,
            matrix_instr_nonkdim=16,
            kpack=1,
        )
        return output.reshape(*x.shape[:-1], _QWEN35_INTERMEDIATE_SIZE)

    gate_up = rocm_unquantized_gemm_impl(x, weight)
    output = x.new_empty((*x.shape[:-1], _QWEN35_INTERMEDIATE_SIZE))
    torch.ops._C.silu_and_mul(output, gate_up)
    return output


def qwen35_bf16_gateup_silu_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0] // 2))


direct_register_custom_op(
    op_name="qwen35_bf16_gateup_silu",
    op_func=qwen35_bf16_gateup_silu_impl,
    fake_impl=qwen35_bf16_gateup_silu_fake,
)


def qwen35_bf16_gateup_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.vllm.qwen35_bf16_gateup_silu(x, weight)
