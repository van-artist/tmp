"""Bitwise and latency check for fused Qwen3.5 continuation KV staging.

Run on the competition gfx936 worker after installing or overlaying this tree::

    python benchmarks/kernels/benchmark_qwen35_continuation_stage.py \
        --warmup 12 --repeats 51

For every case, the reference first runs vLLM's ordinary BF16 cache-update
kernel and then the retained 784-page gather. The candidate updates the same
physical cache and constructs the same dense K/V input in one launch. The
script requires both the complete cache allocation and complete dense prefix
to be bitwise equal before reporting latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from vllm.v1.attention.ops.triton_qwen35_kv_gather import (
    gather_qwen35_paged_kv,
    stage_qwen35_continuation_kv,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)


PAGE_SIZE = 784
NUM_KV_HEADS = 4
HEAD_SIZE = 256
VALUE_TOKEN_STRIDE = 14336
DENSE_GUARD_TOKENS = 17
DEFAULT_CASES = (
    (4096, 2, 62),
    (4096, 16, 48),
    (8192, 511, 1),
    (8192, 783, 1),
    # A synthetic exact physical-page boundary in addition to real 4K chunks.
    (7840, 784, 0),
    (16384, 784, 0),
    (16384, 785, 15),
    # Realistic 4K--8K tail: the final 616 static rows are padding slots.
    (4096, 3480, 616),
    (4096, 4096, 0),
    (12288, 4096, 0),
    (24576, 4096, 0),
    (28672, 4096, 0),
)


def median_ms(fn: Callable[[], None], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return statistics.median(samples)


def parse_case(value: str) -> tuple[int, int, int]:
    fields = value.split(":")
    if len(fields) not in (2, 3):
        raise argparse.ArgumentTypeError("case must be PREFIX:QUERY[:PADDING]")
    prefix, query = (int(field) for field in fields[:2])
    padding = int(fields[2]) if len(fields) == 3 else 0
    if prefix <= 0 or query <= 1 or padding < 0:
        raise argparse.ArgumentTypeError("prefix/query must be positive")
    return prefix, query, padding


def run_case(
    prefix_tokens: int,
    query_tokens: int,
    padding_tokens: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, int | float | bool]:
    device = torch.device("cuda")
    total_tokens = prefix_tokens + query_tokens
    logical_pages = (total_tokens + PAGE_SIZE - 1) // PAGE_SIZE
    num_physical_pages = logical_pages + 5
    generator = torch.Generator(device=device).manual_seed(seed)

    physical_pages = torch.randperm(
        num_physical_pages,
        device=device,
        dtype=torch.int64,
        generator=generator,
    )[:logical_pages]
    block_table = physical_pages.to(torch.int32).unsqueeze(0)

    initial_cache = torch.randn(
        (
            num_physical_pages,
            2,
            PAGE_SIZE,
            NUM_KV_HEADS,
            HEAD_SIZE,
        ),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    reference_cache = initial_cache.clone()
    candidate_cache = initial_cache.clone()
    del initial_cache

    padded_query_tokens = query_tokens + padding_tokens
    key = torch.randn(
        (padded_query_tokens, NUM_KV_HEADS, HEAD_SIZE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    # Match the real Qwen projection: V is the final 1024-element view of a
    # 14336-wide Q/G/K/V output and is not contiguous across token rows.
    value_storage = torch.randn(
        (padded_query_tokens, VALUE_TOKEN_STRIDE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    value = value_storage[:, -NUM_KV_HEADS * HEAD_SIZE :].view(
        padded_query_tokens, NUM_KV_HEADS, HEAD_SIZE
    )
    assert value.stride(0) == VALUE_TOKEN_STRIDE

    logical_current = torch.arange(
        prefix_tokens,
        total_tokens,
        device=device,
        dtype=torch.int64,
    )
    current_pages = block_table[0, logical_current // PAGE_SIZE].to(torch.int64)
    current_slots = (
        current_pages * PAGE_SIZE + logical_current % PAGE_SIZE
    )
    slot_mapping = torch.full(
        (padded_query_tokens,),
        -1,
        device=device,
        dtype=torch.int64,
    )
    slot_mapping[:query_tokens].copy_(current_slots)

    dense_sentinel = 123.0
    reference_dense_storage = torch.full(
        (
            2,
            total_tokens + DENSE_GUARD_TOKENS,
            NUM_KV_HEADS,
            HEAD_SIZE,
        ),
        dense_sentinel,
        device=device,
        dtype=torch.bfloat16,
    )
    candidate_dense_storage = reference_dense_storage.clone()
    reference_dense = reference_dense_storage[:, :total_tokens]
    candidate_dense = candidate_dense_storage[:, :total_tokens]
    reference_dense_key, reference_dense_value = reference_dense.unbind(0)
    candidate_dense_key, candidate_dense_value = candidate_dense.unbind(0)
    reference_key_cache, reference_value_cache = reference_cache.unbind(1)
    candidate_key_cache, candidate_value_cache = candidate_cache.unbind(1)
    one = torch.ones((), device=device, dtype=torch.float32)

    def reference() -> None:
        triton_reshape_and_cache_flash(
            key,
            value,
            reference_key_cache,
            reference_value_cache,
            slot_mapping,
            "auto",
            one,
            one,
        )
        gather_qwen35_paged_kv(
            reference_key_cache,
            reference_value_cache,
            reference_dense_key,
            reference_dense_value,
            block_table,
            total_tokens,
        )

    def candidate(num_warps: int = 4) -> None:
        stage_qwen35_continuation_kv(
            key,
            value,
            candidate_key_cache,
            candidate_value_cache,
            candidate_dense_key,
            candidate_dense_value,
            block_table,
            slot_mapping,
            prefix_tokens,
            query_tokens,
            num_warps=num_warps,
        )

    reference()
    candidate(4)
    torch.cuda.synchronize()
    cache_bitwise = torch.equal(candidate_cache, reference_cache)
    dense_key_bitwise = torch.equal(candidate_dense_key, reference_dense_key)
    dense_value_bitwise = torch.equal(candidate_dense_value, reference_dense_value)
    dense_tail_untouched = bool(
        torch.all(
            candidate_dense_storage[:, total_tokens:] == dense_sentinel
        ).item()
    )
    if not (
        cache_bitwise
        and dense_key_bitwise
        and dense_value_bitwise
        and dense_tail_untouched
    ):
        raise AssertionError(
            "fused staging differs from cache-update plus full gather: "
            f"cache={cache_bitwise}, dense_key={dense_key_bitwise}, "
            f"dense_value={dense_value_bitwise}, "
            f"dense_tail={dense_tail_untouched}"
        )

    # The production launch defaults to four warps, but keep the alternative
    # launch geometry honest before comparing its latency on the target GPU.
    candidate(8)
    torch.cuda.synchronize()
    if not (
        torch.equal(candidate_cache, reference_cache)
        and torch.equal(candidate_dense_key, reference_dense_key)
        and torch.equal(candidate_dense_value, reference_dense_value)
        and bool(
            torch.all(
                candidate_dense_storage[:, total_tokens:] == dense_sentinel
            ).item()
        )
    ):
        raise AssertionError("eight-warp fused staging is not bitwise equal")

    reference_ms = median_ms(reference, warmup, repeats)
    candidate_w4_ms = median_ms(lambda: candidate(4), warmup, repeats)
    candidate_w8_ms = median_ms(lambda: candidate(8), warmup, repeats)
    candidate_ms = min(candidate_w4_ms, candidate_w8_ms)
    return {
        "prefix_tokens": prefix_tokens,
        "query_tokens": query_tokens,
        "padding_tokens": padding_tokens,
        "total_tokens": total_tokens,
        "physical_pages": num_physical_pages,
        "cache_bitwise": cache_bitwise,
        "dense_key_bitwise": dense_key_bitwise,
        "dense_value_bitwise": dense_value_bitwise,
        "dense_tail_untouched": dense_tail_untouched,
        "key_token_stride": key.stride(0),
        "value_token_stride": value.stride(0),
        "reference_update_plus_gather_ms": reference_ms,
        "fused_update_and_stage_w4_ms": candidate_w4_ms,
        "fused_update_and_stage_w8_ms": candidate_w8_ms,
        "best_num_warps": 4 if candidate_w4_ms <= candidate_w8_ms else 8,
        "best_fused_update_and_stage_ms": candidate_ms,
        "speedup": reference_ms / candidate_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=51)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        type=parse_case,
        help="PREFIX:QUERY[:PADDING]; repeat to benchmark multiple cases",
    )
    args = parser.parse_args()
    assert torch.cuda.is_available()

    cases = tuple(args.cases) if args.cases else DEFAULT_CASES
    for case_index, (prefix, query, padding) in enumerate(cases):
        result = run_case(
            prefix,
            query,
            padding,
            args.warmup,
            args.repeats,
            args.seed + case_index,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
