# SPDX-License-Identifier: Apache-2.0
"""Standalone benchmark for fused Qwen3.5 solve_tril + recompute W/U."""

import argparse

import torch
from triton.testing import do_bench

from vllm.model_executor.layers.fla.ops.aiter_gdn_opt import (
    fused_solve_tril_recompute,
)

fused_solve_tril_recompute_w_u = (
    fused_solve_tril_recompute.fused_solve_tril_recompute_w_u
)
from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd

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
    # chunk_local_cumsum produces non-positive cumulative log gates.  A smooth
    # synthetic range keeps the triangular system representative and stable.
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


def baseline(a_raw, k, v, beta, g):
    inverse = solve_tril(a_raw, cu_seqlens=None, output_dtype=k.dtype)
    return recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g,
        A=inverse,
        cu_seqlens=None,
    )


def fused(a_raw, k, v, beta, g):
    # The AITER-derived kernel returns head-major [B,H,T,D] outputs.
    w, u = fused_solve_tril_recompute_w_u(
        A_raw=a_raw,
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g,
        cu_seqlens=None,
    )
    return w.permute(0, 2, 1, 3), u.permute(0, 2, 1, 3)


def report_diff(name: str, actual: torch.Tensor, expected: torch.Tensor):
    diff = (actual.float() - expected.float()).abs()
    print(
        f"{name}: max_abs={diff.max().item():.8f}, "
        f"mean_abs={diff.mean().item():.8f}"
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


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
    print("JIT compiling and checking W/U...")
    reference_w, reference_u = baseline(a_raw, k, v, beta, g)
    fused_w, fused_u = fused(a_raw, k, v, beta, g)
    torch.cuda.synchronize()
    report_diff("W", fused_w, reference_w)
    report_diff("U", fused_u, reference_u)
    print("correctness=PASS")

    if args.correctness_only:
        return

    baseline_ms = do_bench(lambda: baseline(a_raw, k, v, beta, g))
    fused_ms = do_bench(lambda: fused(a_raw, k, v, beta, g))
    print(f"solve_plus_recompute={baseline_ms * 1000:.2f} us")
    print(
        f"fused_solve_recompute={fused_ms * 1000:.2f} us, "
        f"speedup={baseline_ms / fused_ms:.3f}x"
    )
    print(
        f"estimated_48_layer_saving="
        f"{(baseline_ms - fused_ms) * 48:.3f} ms/chunk"
    )


if __name__ == "__main__":
    main()
