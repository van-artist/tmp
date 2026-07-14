# SPDX-License-Identifier: Apache-2.0
"""Standalone old-vs-new solve_tril allocation/write benchmark."""

import argparse

import torch
from triton.testing import do_bench

from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from vllm.model_executor.layers.fla.ops.solve_tril import (
    FLA_TRIL_PRECISION,
    merge_16x16_to_64x64_inverse_kernel,
    solve_tril,
)
from vllm.model_executor.layers.fla.ops.utils import is_tma_supported
from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd
from vllm.triton_utils import triton

BT = 64
NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_SIZE = 128


def make_inputs(tokens: int, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    k = torch.randn(
        (1, tokens, NUM_K_HEADS, HEAD_SIZE),
        device=device,
        dtype=torch.bfloat16,
    ) * 0.05
    v = torch.randn(
        (1, tokens, NUM_V_HEADS, HEAD_SIZE),
        device=device,
        dtype=torch.bfloat16,
    ) * 0.05
    beta = torch.sigmoid(
        torch.randn(
            (1, tokens, NUM_V_HEADS), device=device, dtype=torch.float32
        )
    )
    gate_step = -torch.rand(
        (1, tokens, NUM_V_HEADS), device=device, dtype=torch.float32
    ) * 0.05
    g = gate_step.reshape(1, tokens // BT, BT, NUM_V_HEADS).cumsum(dim=2)
    g = g.reshape(1, tokens, NUM_V_HEADS).contiguous()
    a_raw = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=None,
        chunk_size=BT,
        output_dtype=torch.float32,
    )
    return a_raw, k, v, beta, g


def legacy_zeroed(a_raw: torch.Tensor):
    """Original behavior: full memset, kernel leaves upper blocks untouched."""
    batch, tokens, heads, block = a_raw.shape
    output = torch.zeros_like(a_raw, dtype=torch.bfloat16)
    merge_16x16_to_64x64_inverse_kernel[
        (triton.cdiv(tokens, block), batch * heads)
    ](
        A=a_raw,
        Ai=output,
        cu_seqlens=None,
        chunk_indices=None,
        T=tokens,
        H=heads,
        BT=block,
        USE_TMA=is_tma_supported,
        DOT_PRECISION=FLA_TRIL_PRECISION,
        WRITE_UPPER=False,
    )
    return output


def optimized_empty(a_raw: torch.Tensor):
    return solve_tril(a_raw, cu_seqlens=None, output_dtype=torch.bfloat16)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    if args.tokens <= 0 or args.tokens % BT:
        raise ValueError("--tokens must be a positive multiple of 64")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is not available")
    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", "unknown"))
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"arch={arch}")
    if arch.split(":", 1)[0] != "gfx936":
        raise RuntimeError(f"expected gfx936, got {arch}")

    print(f"building exact GDN tensors for T={args.tokens}...")
    a_raw, k, v, beta, g = make_inputs(args.tokens, args.seed)
    print("JIT compiling and comparing inverse/W/U...")
    old_inverse = legacy_zeroed(a_raw)
    new_inverse = optimized_empty(a_raw)
    torch.cuda.synchronize()
    torch.testing.assert_close(new_inverse, old_inverse, atol=0, rtol=0)

    local_row = torch.arange(args.tokens, device="cuda") % BT
    upper_mask = torch.arange(BT, device="cuda")[None, :] > local_row[:, None]
    upper = new_inverse[0].permute(1, 0, 2)[:, upper_mask]
    if torch.count_nonzero(upper).item() != 0:
        raise AssertionError("optimized solve left nonzero upper-triangle values")

    old_w, old_u = recompute_w_u_fwd(k, v, beta, g, old_inverse, None)
    new_w, new_u = recompute_w_u_fwd(k, v, beta, g, new_inverse, None)
    torch.cuda.synchronize()
    torch.testing.assert_close(new_w, old_w, atol=0, rtol=0)
    torch.testing.assert_close(new_u, old_u, atol=0, rtol=0)
    print("inverse/W/U correctness=PASS")

    if args.correctness_only:
        return

    old_ms = do_bench(lambda: legacy_zeroed(a_raw))
    new_ms = do_bench(lambda: optimized_empty(a_raw))
    print(f"legacy_zeros_solve={old_ms * 1000:.2f} us")
    print(f"empty_full_write_solve={new_ms * 1000:.2f} us")
    print(f"speedup={old_ms / new_ms:.3f}x")
    print(
        f"estimated_48_layer_saving={(old_ms - new_ms) * 48:.3f} ms/chunk"
    )


if __name__ == "__main__":
    main()
