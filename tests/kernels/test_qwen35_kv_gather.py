# SPDX-License-Identifier: Apache-2.0
"""Correctness and opt-in benchmarks for the Qwen3.5 paged-KV gather."""

import os

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_qwen35_kv_gather import (
    gather_qwen35_paged_kv,
    gather_qwen35_paged_kv_chunked,
)

PAGE_SIZE = 784
NUM_KV_HEADS = 4
HEAD_SIZE = 256
ROW_SIZE = NUM_KV_HEADS * HEAD_SIZE

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Qwen3.5 gather specialization is ROCm-only"
)


def _make_inputs(total_tokens: int):
    logical_pages = (total_tokens + PAGE_SIZE - 1) // PAGE_SIZE
    # Leave holes and reverse the mapping so an accidentally logical/contiguous
    # page lookup cannot pass the comparison.
    physical_pages = logical_pages + 3
    block_table = torch.arange(
        logical_pages - 1, -1, -1, device="cuda", dtype=torch.int32
    )
    block_table = block_table + 2
    cache = torch.randn(
        physical_pages,
        2,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    # Unbinding reproduces the doubled block stride of vLLM's interleaved K/V
    # allocation, which is essential to exercise the production addressing.
    key_cache, value_cache = cache.unbind(1)
    return key_cache, value_cache, block_table


@pytest.mark.parametrize("total_tokens", [1, 783, 784, 785, 4096, 32768])
@pytest.mark.parametrize("tokens_per_program", [4, 8, 16])
@pytest.mark.parametrize("num_warps", [2, 4, 8])
def test_chunked_gather_matches_row_kernel(
    total_tokens: int, tokens_per_program: int, num_warps: int
):
    key_cache, value_cache, block_table = _make_inputs(total_tokens)
    output_shape = (total_tokens + 7, NUM_KV_HEADS, HEAD_SIZE)
    ref_key = torch.full(
        output_shape, float("nan"), device="cuda", dtype=torch.bfloat16
    )
    ref_value = torch.full_like(ref_key, float("nan"))
    actual_key = torch.full_like(ref_key, float("nan"))
    actual_value = torch.full_like(ref_key, float("nan"))

    gather_qwen35_paged_kv(
        key_cache,
        value_cache,
        ref_key,
        ref_value,
        block_table,
        total_tokens,
    )
    gather_qwen35_paged_kv_chunked(
        key_cache,
        value_cache,
        actual_key,
        actual_value,
        block_table,
        total_tokens,
        tokens_per_program=tokens_per_program,
        num_warps=num_warps,
    )

    physical_pages = block_table.to(torch.long)
    expected_key = key_cache.index_select(0, physical_pages).flatten(0, 1)
    expected_value = value_cache.index_select(0, physical_pages).flatten(0, 1)
    torch.testing.assert_close(
        ref_key[:total_tokens], expected_key[:total_tokens], rtol=0, atol=0
    )
    torch.testing.assert_close(
        ref_value[:total_tokens], expected_value[:total_tokens], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_key[:total_tokens], ref_key[:total_tokens], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_value[:total_tokens], ref_value[:total_tokens], rtol=0, atol=0
    )
    assert torch.isnan(actual_key[total_tokens:]).all()
    assert torch.isnan(actual_value[total_tokens:]).all()


@pytest.mark.skipif(
    os.getenv("VLLM_RUN_QWEN35_GATHER_BENCHMARK") != "1",
    reason="set VLLM_RUN_QWEN35_GATHER_BENCHMARK=1 to run timing comparison",
)
@pytest.mark.parametrize("total_tokens", [4096, 8192, 16384, 32768])
def test_benchmark_chunked_gather(total_tokens: int):
    from triton.testing import do_bench

    key_cache, value_cache, block_table = _make_inputs(total_tokens)
    dense_key = torch.empty(
        total_tokens, NUM_KV_HEADS, HEAD_SIZE, device="cuda", dtype=torch.bfloat16
    )
    dense_value = torch.empty_like(dense_key)

    baseline_ms = do_bench(
        lambda: gather_qwen35_paged_kv(
            key_cache,
            value_cache,
            dense_key,
            dense_value,
            block_table,
            total_tokens,
        )
    )
    # Required traffic: read and write both K and V.
    traffic_gib = total_tokens * ROW_SIZE * 2 * 4 / 2**30
    print(
        f"tokens={total_tokens} row: {baseline_ms * 1000:.2f} us "
        f"{traffic_gib / (baseline_ms / 1000):.2f} GiB/s"
    )

    for tokens_per_program in (4, 8, 16):
        for num_warps in (2, 4, 8):
            candidate_ms = do_bench(
                lambda: gather_qwen35_paged_kv_chunked(
                    key_cache,
                    value_cache,
                    dense_key,
                    dense_value,
                    block_table,
                    total_tokens,
                    tokens_per_program=tokens_per_program,
                    num_warps=num_warps,
                )
            )
            print(
                f"tokens={total_tokens} tpp={tokens_per_program} "
                f"warps={num_warps}: {candidate_ms * 1000:.2f} us "
                f"{traffic_gib / (candidate_ms / 1000):.2f} GiB/s "
                f"speedup={baseline_ms / candidate_ms:.3f}x"
            )
