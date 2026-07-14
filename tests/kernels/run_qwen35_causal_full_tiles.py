# SPDX-License-Identifier: Apache-2.0
"""Standalone original-vs-full-history-tile Qwen3.5 attention benchmark."""

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


def run_attention(
    inputs, output: torch.Tensor, kv_tokens: int, fast_full_tiles: bool
) -> None:
    query, key, value, block_table, cu_seqlens_q, seqused_k = inputs
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
        alibi_slopes=None,
        output_scale=None,
        qq_bias=None,
        sinks=None,
        mm_prefix_range=None,
        gfx936_qwen35_prefill=True,
        gfx936_causal_full_tiles=fast_full_tiles,
    )


@torch.inference_mode()
def run_case(kv_tokens: int, seed: int) -> None:
    if kv_tokens < QUERY_TOKENS:
        raise ValueError("kv_tokens must be at least the 4096-token query chunk")
    print(
        f"\ncase: query_tokens={QUERY_TOKENS}, kv_tokens={kv_tokens}, "
        f"context_tokens={kv_tokens - QUERY_TOKENS}"
    )
    inputs = make_inputs(kv_tokens, seed)
    baseline = torch.empty_like(inputs[0])
    candidate = torch.empty_like(inputs[0])
    print("JIT compiling baseline and causal-full-tile variants...")
    run_attention(inputs, baseline, kv_tokens, fast_full_tiles=False)
    run_attention(inputs, candidate, kv_tokens, fast_full_tiles=True)
    torch.cuda.synchronize()

    diff = (baseline.float() - candidate.float()).abs()
    print(
        f"max_abs={diff.max().item():.8f}, "
        f"mean_abs={diff.mean().item():.8f}"
    )
    torch.testing.assert_close(candidate, baseline, atol=0, rtol=0)
    print("correctness=PASS")

    baseline_ms = do_bench(
        lambda: run_attention(
            inputs, baseline, kv_tokens, fast_full_tiles=False
        )
    )
    candidate_ms = do_bench(
        lambda: run_attention(
            inputs, candidate, kv_tokens, fast_full_tiles=True
        )
    )
    print(f"baseline_all_tiles_masked={baseline_ms:.3f} ms")
    print(f"causal_full_tile_fast={candidate_ms:.3f} ms")
    print(f"speedup={baseline_ms / candidate_ms:.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kv-tokens", type=int, nargs="+", default=[4096, 16384, 32768]
    )
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is not available")
    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", "unknown"))
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"arch={arch}")
    if arch.split(":", 1)[0] != "gfx936":
        raise RuntimeError(f"expected gfx936, got {arch}")
    for kv_tokens in args.kv_tokens:
        run_case(kv_tokens, args.seed)


if __name__ == "__main__":
    main()
