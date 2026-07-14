# SPDX-License-Identifier: Apache-2.0
"""Standalone Qwen3.5 KV page-layout experiments for gfx936.

This runner intentionally does not import pytest.  It compares the current
one-token-row-per-program gather with page-major gathers and with a contiguous
physical-page fast path.  No production dispatch is changed.
"""

import argparse

import torch
from triton.testing import do_bench

from vllm.triton_utils import tl, triton


KV_HEADS = 4
HEAD_SIZE = 256
ROW_SIZE = KV_HEADS * HEAD_SIZE


@triton.jit
def _row_gather_kernel(
    key_cache,
    value_cache,
    dense_key,
    dense_value,
    block_table,
    total_elements,
    stride_k_block,
    stride_v_block,
    PAGE_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
):
    """Production baseline: one token row per program."""
    in_page_token = tl.program_id(0)
    logical_page = tl.program_id(1)
    within = in_page_token * ROW_SIZE + tl.arange(0, ROW_SIZE)
    global_offset = logical_page * PAGE_SIZE * ROW_SIZE + within
    mask = global_offset < total_elements
    physical_page = tl.load(block_table + logical_page).to(tl.int64)
    key = tl.load(
        key_cache + physical_page * stride_k_block + within,
        mask=mask,
        other=0.0,
    )
    value = tl.load(
        value_cache + physical_page * stride_v_block + within,
        mask=mask,
        other=0.0,
    )
    tl.store(dense_key + global_offset, key, mask=mask)
    tl.store(dense_value + global_offset, value, mask=mask)


@triton.jit
def _page_major_gather_kernel(
    key_cache,
    value_cache,
    dense_key,
    dense_value,
    block_table,
    total_tokens,
    stride_k_block,
    stride_v_block,
    PAGE_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
):
    """Copy several token rows after loading one page-table entry."""
    token_group = tl.program_id(0)
    logical_page = tl.program_id(1)
    token_offsets = (
        token_group * TOKENS_PER_PROGRAM
        + tl.arange(0, TOKENS_PER_PROGRAM)
    )
    element_offsets = tl.arange(0, ROW_SIZE)
    global_tokens = logical_page * PAGE_SIZE + token_offsets
    mask = (token_offsets[:, None] < PAGE_SIZE) & (
        global_tokens[:, None] < total_tokens
    )
    within = token_offsets[:, None] * ROW_SIZE + element_offsets[None, :]
    dense_offsets = global_tokens[:, None] * ROW_SIZE + element_offsets[None, :]
    physical_page = tl.load(block_table + logical_page).to(tl.int64)
    key = tl.load(
        key_cache + physical_page * stride_k_block + within,
        mask=mask,
        other=0.0,
    )
    value = tl.load(
        value_cache + physical_page * stride_v_block + within,
        mask=mask,
        other=0.0,
    )
    tl.store(dense_key + dense_offsets, key, mask=mask)
    tl.store(dense_value + dense_offsets, value, mask=mask)


@triton.jit
def _page_major_loop_gather_kernel(
    key_cache,
    value_cache,
    dense_key,
    dense_value,
    block_table,
    total_tokens,
    stride_k_block,
    stride_v_block,
    PAGE_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
):
    """Reuse one page-table load while copying one row at a time.

    Unlike ``_page_major_gather_kernel``, this does not construct a
    [TOKENS_PER_PROGRAM, ROW_SIZE] tile.  Only one 1024-element K/V row is
    live in each loop iteration, reducing VGPR pressure on wave64 GPUs.
    """
    token_group = tl.program_id(0)
    logical_page = tl.program_id(1)
    element_offsets = tl.arange(0, ROW_SIZE)
    physical_page = tl.load(block_table + logical_page).to(tl.int64)
    page_input_base = physical_page * stride_k_block
    page_value_base = physical_page * stride_v_block
    logical_token_base = logical_page * PAGE_SIZE

    for token_delta in tl.range(
        0, TOKENS_PER_PROGRAM, loop_unroll_factor=1
    ):
        in_page_token = token_group * TOKENS_PER_PROGRAM + token_delta
        global_token = logical_token_base + in_page_token
        mask = (in_page_token < PAGE_SIZE) & (global_token < total_tokens)
        within = in_page_token * ROW_SIZE + element_offsets
        dense_offsets = global_token * ROW_SIZE + element_offsets
        key = tl.load(
            key_cache + page_input_base + within,
            mask=mask,
            other=0.0,
        )
        tl.store(dense_key + dense_offsets, key, mask=mask)
        value = tl.load(
            value_cache + page_value_base + within,
            mask=mask,
            other=0.0,
        )
        tl.store(dense_value + dense_offsets, value, mask=mask)


@triton.jit
def _contiguous_page_gather_kernel(
    key_cache,
    value_cache,
    dense_key,
    dense_value,
    total_tokens,
    base_page,
    stride_k_block,
    stride_v_block,
    PAGE_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
):
    """Page-major gather using base_page + logical_page, without a table."""
    token_group = tl.program_id(0)
    logical_page = tl.program_id(1)
    token_offsets = (
        token_group * TOKENS_PER_PROGRAM
        + tl.arange(0, TOKENS_PER_PROGRAM)
    )
    element_offsets = tl.arange(0, ROW_SIZE)
    global_tokens = logical_page * PAGE_SIZE + token_offsets
    mask = (token_offsets[:, None] < PAGE_SIZE) & (
        global_tokens[:, None] < total_tokens
    )
    within = token_offsets[:, None] * ROW_SIZE + element_offsets[None, :]
    dense_offsets = global_tokens[:, None] * ROW_SIZE + element_offsets[None, :]
    physical_page = base_page + logical_page
    key = tl.load(
        key_cache + physical_page * stride_k_block + within,
        mask=mask,
        other=0.0,
    )
    value = tl.load(
        value_cache + physical_page * stride_v_block + within,
        mask=mask,
        other=0.0,
    )
    tl.store(dense_key + dense_offsets, key, mask=mask)
    tl.store(dense_value + dense_offsets, value, mask=mask)


def _launch_row(cache, dense, table, tokens, page_size):
    key_cache, value_cache = cache
    dense_key, dense_value = dense
    _row_gather_kernel[(page_size, triton.cdiv(tokens, page_size))](
        key_cache,
        value_cache,
        dense_key,
        dense_value,
        table,
        tokens * ROW_SIZE,
        key_cache.stride(0),
        value_cache.stride(0),
        PAGE_SIZE=page_size,
        ROW_SIZE=ROW_SIZE,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )


def _launch_page(cache, dense, table, tokens, page_size, tpp, warps):
    key_cache, value_cache = cache
    dense_key, dense_value = dense
    grid = (triton.cdiv(page_size, tpp), triton.cdiv(tokens, page_size))
    _page_major_gather_kernel[grid](
        key_cache,
        value_cache,
        dense_key,
        dense_value,
        table,
        tokens,
        key_cache.stride(0),
        value_cache.stride(0),
        PAGE_SIZE=page_size,
        ROW_SIZE=ROW_SIZE,
        TOKENS_PER_PROGRAM=tpp,
        num_warps=warps,
        num_stages=1,
        waves_per_eu=1,
    )


def _launch_page_loop(cache, dense, table, tokens, page_size, tpp, warps):
    key_cache, value_cache = cache
    dense_key, dense_value = dense
    grid = (triton.cdiv(page_size, tpp), triton.cdiv(tokens, page_size))
    _page_major_loop_gather_kernel[grid](
        key_cache,
        value_cache,
        dense_key,
        dense_value,
        table,
        tokens,
        key_cache.stride(0),
        value_cache.stride(0),
        PAGE_SIZE=page_size,
        ROW_SIZE=ROW_SIZE,
        TOKENS_PER_PROGRAM=tpp,
        num_warps=warps,
        num_stages=1,
        waves_per_eu=1,
    )


def _launch_contiguous(cache, dense, tokens, page_size, tpp, warps):
    key_cache, value_cache = cache
    dense_key, dense_value = dense
    grid = (triton.cdiv(page_size, tpp), triton.cdiv(tokens, page_size))
    _contiguous_page_gather_kernel[grid](
        key_cache,
        value_cache,
        dense_key,
        dense_value,
        tokens,
        0,
        key_cache.stride(0),
        value_cache.stride(0),
        PAGE_SIZE=page_size,
        ROW_SIZE=ROW_SIZE,
        TOKENS_PER_PROGRAM=tpp,
        num_warps=warps,
        num_stages=1,
        waves_per_eu=1,
    )


def _bandwidth_gbps(tokens: int, milliseconds: float) -> float:
    # K and V are each read once and written once.  ROW_SIZE BF16 elements
    # per tensor gives 8 * ROW_SIZE bytes of total HBM traffic per token.
    traffic_bytes = tokens * 8 * ROW_SIZE
    return traffic_bytes / (milliseconds * 1e6)


@torch.inference_mode()
def run_case(tokens: int, page_size: int) -> None:
    num_pages = triton.cdiv(tokens, page_size)
    print(f"\ncase: tokens={tokens}, page_size={page_size}, pages={num_pages}")
    key_cache = torch.randn(
        num_pages,
        page_size,
        KV_HEADS,
        HEAD_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value_cache = torch.randn_like(key_cache)
    dense_key = torch.empty(
        tokens, KV_HEADS, HEAD_SIZE, device="cuda", dtype=torch.bfloat16
    )
    dense_value = torch.empty_like(dense_key)
    reference_key = torch.empty_like(dense_key)
    reference_value = torch.empty_like(dense_value)

    # Random page order verifies the table-based kernels.  A separate identity
    # table is used for the contiguous arithmetic fast path.
    random_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32)
    identity_table = torch.arange(num_pages, device="cuda", dtype=torch.int32)
    cache = (key_cache, value_cache)
    reference = (reference_key, reference_value)
    output = (dense_key, dense_value)

    _launch_row(cache, reference, random_table, tokens, page_size)
    torch.cuda.synchronize()

    row_ms = do_bench(
        lambda: _launch_row(cache, output, random_table, tokens, page_size)
    )
    print(
        f"row/table: {row_ms:.4f} ms, "
        f"{_bandwidth_gbps(tokens, row_ms):.1f} GB/s"
    )

    for tpp in (4, 8, 16):
        for warps in (2, 4, 8):
            _launch_page(
                cache, output, random_table, tokens, page_size, tpp, warps
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(dense_key, reference_key, atol=0, rtol=0)
            torch.testing.assert_close(
                dense_value, reference_value, atol=0, rtol=0
            )
            elapsed = do_bench(
                lambda tpp=tpp, warps=warps: _launch_page(
                    cache,
                    output,
                    random_table,
                    tokens,
                    page_size,
                    tpp,
                    warps,
                )
            )
            print(
                f"page/table tpp={tpp:2d} warps={warps}: "
                f"{elapsed:.4f} ms, {_bandwidth_gbps(tokens, elapsed):.1f} "
                f"GB/s, {row_ms / elapsed:.3f}x"
            )

    print("low-live-range page loop:")
    for tpp in (4, 8, 16):
        for warps in (2, 4, 8):
            _launch_page_loop(
                cache, output, random_table, tokens, page_size, tpp, warps
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(dense_key, reference_key, atol=0, rtol=0)
            torch.testing.assert_close(
                dense_value, reference_value, atol=0, rtol=0
            )
            elapsed = do_bench(
                lambda tpp=tpp, warps=warps: _launch_page_loop(
                    cache,
                    output,
                    random_table,
                    tokens,
                    page_size,
                    tpp,
                    warps,
                )
            )
            print(
                f"page-loop  tpp={tpp:2d} warps={warps}: "
                f"{elapsed:.4f} ms, {_bandwidth_gbps(tokens, elapsed):.1f} "
                f"GB/s, {row_ms / elapsed:.3f}x"
            )

    # Establish an identity-table reference before testing base+logical_page.
    _launch_row(cache, reference, identity_table, tokens, page_size)
    _launch_contiguous(cache, output, tokens, page_size, 8, 4)
    torch.cuda.synchronize()
    torch.testing.assert_close(dense_key, reference_key, atol=0, rtol=0)
    torch.testing.assert_close(dense_value, reference_value, atol=0, rtol=0)
    table_ms = do_bench(
        lambda: _launch_page(
            cache, output, identity_table, tokens, page_size, 8, 4
        )
    )
    contiguous_ms = do_bench(
        lambda: _launch_contiguous(cache, output, tokens, page_size, 8, 4)
    )
    print(
        f"contiguous/base tpp= 8 warps=4: {contiguous_ms:.4f} ms, "
        f"{_bandwidth_gbps(tokens, contiguous_ms):.1f} GB/s, "
        f"vs_identity_table={table_ms / contiguous_ms:.3f}x"
    )
    print("correctness=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[4096, 16384, 32768]
    )
    parser.add_argument(
        "--page-sizes", type=int, nargs="+", default=[784, 832]
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is not available")
    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", "unknown"))
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"arch={arch}")
    if arch.split(":", 1)[0] != "gfx936":
        raise RuntimeError(f"expected gfx936, got {arch}")
    for page_size in args.page_sizes:
        if page_size <= 0 or page_size % 16:
            raise ValueError("page sizes must be positive multiples of 16")
        for tokens in args.tokens:
            if tokens <= 0:
                raise ValueError("tokens must be positive")
            run_case(tokens, page_size)


if __name__ == "__main__":
    main()
