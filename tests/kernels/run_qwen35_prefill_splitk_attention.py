# SPDX-License-Identifier: Apache-2.0
"""Standalone 2D-vs-3D split-K Qwen3.5 long-prefill benchmark."""

import argparse

import torch
from triton.testing import do_bench

from vllm.v1.attention.ops.triton_unified_attention import unified_attention


BLOCK_SIZE = 784
QUERY_TOKENS = 4096
QUERY_HEADS = 24
KV_HEADS = 4
HEAD_SIZE = 256


def make_inputs(kv_tokens: int, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    pages = (kv_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    query = torch.randn(
        QUERY_TOKENS,
        QUERY_HEADS,
        HEAD_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        pages,
        BLOCK_SIZE,
        KV_HEADS,
        HEAD_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    block_table = torch.randperm(
        pages, device=device, dtype=torch.int32
    ).unsqueeze(0)
    cu_seqlens_q = torch.tensor(
        [0, QUERY_TOKENS], device=device, dtype=torch.int32
    )
    seqused_k = torch.tensor([kv_tokens], device=device, dtype=torch.int32)
    return query, key, value, block_table, cu_seqlens_q, seqused_k


def make_workspace(query: torch.Tensor, segments: int):
    output = torch.empty(
        QUERY_TOKENS,
        QUERY_HEADS,
        segments,
        HEAD_SIZE,
        device=query.device,
        dtype=torch.float32,
    )
    max_values = torch.empty(
        QUERY_TOKENS,
        QUERY_HEADS,
        segments,
        device=query.device,
        dtype=torch.float32,
    )
    expsum = torch.empty_like(max_values)
    return output, max_values, expsum


def run_attention(
    inputs,
    output: torch.Tensor,
    kv_tokens: int,
    segments: int | None,
    workspace=None,
) -> None:
    query, key, value, block_table, cu_seqlens_q, seqused_k = inputs
    if workspace is None:
        segm_output = segm_max = segm_expsum = None
    else:
        segm_output, segm_max, segm_expsum = workspace
    unified_attention(
        q=query,
        k=key,
        v=value,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=QUERY_TOKENS,
        seqused_k=seqused_k,
        max_seqlen_k=kv_tokens,
        softmax_scale=HEAD_SIZE**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=1 if segments is not None else None,
        num_par_softmax_segments=segments,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
        alibi_slopes=None,
        output_scale=None,
        qq_bias=None,
        sinks=None,
        mm_prefix_range=None,
        gfx936_qwen35_prefill=True,
        gfx936_delay_v_load=False,
        gfx936_force_prefill_3d=segments is not None,
    )


@torch.inference_mode()
def run_case(kv_tokens: int, segment_counts: list[int], seed: int) -> None:
    if kv_tokens < QUERY_TOKENS:
        raise ValueError("kv_tokens must be at least the 4096-token query chunk")
    print(
        f"\ncase: query_tokens={QUERY_TOKENS}, kv_tokens={kv_tokens}, "
        f"context_tokens={kv_tokens - QUERY_TOKENS}"
    )
    inputs = make_inputs(kv_tokens, seed)
    query = inputs[0]
    baseline = torch.empty_like(query)
    print("JIT compiling 2D baseline...")
    run_attention(inputs, baseline, kv_tokens, segments=None)
    torch.cuda.synchronize()
    baseline_ms = do_bench(
        lambda: run_attention(inputs, baseline, kv_tokens, segments=None)
    )
    print(f"baseline_2d={baseline_ms:.3f} ms")

    for segments in segment_counts:
        candidate = torch.empty_like(query)
        workspace = make_workspace(query, segments)
        workspace_mib = sum(x.numel() * x.element_size() for x in workspace) / 2**20
        print(
            f"JIT compiling 3D split-K segments={segments}, "
            f"workspace={workspace_mib:.1f} MiB..."
        )
        run_attention(inputs, candidate, kv_tokens, segments, workspace)
        torch.cuda.synchronize()
        diff = (baseline.float() - candidate.float()).abs()
        print(
            f"segments={segments}: max_abs={diff.max().item():.8f}, "
            f"mean_abs={diff.mean().item():.8f}"
        )
        torch.testing.assert_close(candidate, baseline, atol=0.02, rtol=0.02)
        elapsed = do_bench(
            lambda: run_attention(
                inputs, candidate, kv_tokens, segments, workspace
            )
        )
        print(
            f"splitk_3d_{segments}={elapsed:.3f} ms, "
            f"speedup={baseline_ms / elapsed:.3f}x"
        )
        del candidate, workspace

    print("correctness=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kv-tokens", type=int, nargs="+", default=[16384, 32768]
    )
    parser.add_argument(
        "--segments", type=int, nargs="+", default=[2, 4, 8]
    )
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is not available")
    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", "unknown"))
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"arch={arch}")
    if arch.split(":", 1)[0] != "gfx936":
        raise RuntimeError(f"expected gfx936, got {arch}")
    if any(x <= 0 or x & (x - 1) for x in args.segments):
        raise ValueError("segment counts must be positive powers of two")
    for kv_tokens in args.kv_tokens:
        run_case(kv_tokens, args.segments, args.seed)


if __name__ == "__main__":
    main()
