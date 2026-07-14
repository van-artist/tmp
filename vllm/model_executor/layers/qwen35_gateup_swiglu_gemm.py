# SPDX-License-Identifier: Apache-2.0
"""Experimental gfx936 Qwen3.5 prefill Gate-Up GEMM + SwiGLU epilogue."""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_K = 5120
_N = 17408


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=1,
        ),
    ],
    key=["M", "K", "N"],
)
@triton.jit(do_not_specialize=["M"])
def _qwen35_gateup_swiglu_gemm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    M,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute paired gate/up output tiles and apply SwiGLU before store."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        gate_weight = tl.load(
            weight_ptr
            + offs_n[:, None] * stride_wn
            + k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (k[None, :] < K),
            other=0.0,
        )
        up_weight = tl.load(
            weight_ptr
            + (N + offs_n[:, None]) * stride_wn
            + k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (k[None, :] < K),
            other=0.0,
        )
        gate_acc += tl.dot(
            a,
            tl.trans(gate_weight),
            input_precision="ieee",
        )
        up_acc += tl.dot(
            a,
            tl.trans(up_weight),
            input_precision="ieee",
        )

    # The baseline hipBLAS GEMM stores BF16 gate/up values before the native
    # SwiGLU kernel reads them.  Preserve that precision boundary.
    gate = gate_acc.to(tl.bfloat16).to(tl.float32)
    up = up_acc.to(tl.bfloat16).to(tl.float32)
    activated = gate * tl.sigmoid(gate) * up
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        activated.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def qwen35_gateup_swiglu_gemm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Run the exact Qwen3.5 prefill Gate-Up GEMM with fused SwiGLU."""
    if x.ndim != 2 or x.shape[0] not in (4096, 8192) or x.shape[1] != _K:
        raise ValueError("expected x shape [4096|8192, 5120]")
    if weight.shape != (2 * _N, _K):
        raise ValueError("expected weight shape [34816, 5120]")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("x and weight must be BF16")
    if (
        x.device != weight.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("x and weight must be contiguous on the same device")

    output = x.new_empty((x.shape[0], _N))
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(x.shape[0], meta["BLOCK_M"]),
        triton.cdiv(_N, meta["BLOCK_N"]),
    )
    _qwen35_gateup_swiglu_gemm_kernel[grid](
        x,
        weight,
        output,
        x.shape[0],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        K=_K,
        N=_N,
        waves_per_eu=1,
        matrix_instr_nonkdim=16,
        kpack=2,
    )
    return output


def qwen35_gateup_swiglu_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0] // 2))


direct_register_custom_op(
    op_name="qwen35_gateup_swiglu_gemm",
    op_func=qwen35_gateup_swiglu_gemm_impl,
    fake_impl=qwen35_gateup_swiglu_gemm_fake,
)
