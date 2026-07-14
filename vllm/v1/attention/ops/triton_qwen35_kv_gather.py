# SPDX-License-Identifier: Apache-2.0
"""Fast page-to-dense KV gather for the gfx936 Qwen3.5 prefill path."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_qwen35_nhd_pages_kernel(
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
    """Copy one contiguous [num_kv_heads, head_size] row per program.

    K and V pages are interleaved in vLLM's allocation.  Consequently their
    unbound views have a block stride twice the size of one K or V page.  Use
    the actual view strides instead of assuming tightly packed pages.
    """
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
    tl.store(dense_key + global_offset, key, mask=mask)
    value = tl.load(
        value_cache + physical_page * stride_v_block + within,
        mask=mask,
        other=0.0,
    )
    tl.store(dense_value + global_offset, value, mask=mask)


@triton.jit
def _stage_qwen35_continuation_kv_kernel(
    key,
    value,
    key_cache,
    value_cache,
    dense_key,
    dense_value,
    block_table,
    slot_mapping,
    prefix_tokens,
    total_tokens,
    stride_key_token,
    stride_value_token,
    stride_k_block,
    stride_v_block,
    stride_k_cache_token,
    stride_v_cache_token,
    stride_dense_key_token,
    stride_dense_value_token,
    PAGE_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
):
    """Update current K/V and stage the complete sequence in one launch."""
    in_page_token = tl.program_id(0)
    logical_page = tl.program_id(1)
    logical_token = logical_page * PAGE_SIZE + in_page_token
    if logical_token >= total_tokens:
        return

    offsets = tl.arange(0, ROW_SIZE)
    if logical_token < prefix_tokens:
        physical_page = tl.load(block_table + logical_page).to(tl.int64)
        key_cache_offset = (
            physical_page * stride_k_block
            + in_page_token * stride_k_cache_token
            + offsets
        )
        key_row = tl.load(key_cache + key_cache_offset)
        tl.store(
            dense_key + logical_token * stride_dense_key_token + offsets,
            key_row,
        )
        value_cache_offset = (
            physical_page * stride_v_block
            + in_page_token * stride_v_cache_token
            + offsets
        )
        value_row = tl.load(value_cache + value_cache_offset)
        tl.store(
            dense_value + logical_token * stride_dense_value_token + offsets,
            value_row,
        )
        return

    current_token = logical_token - prefix_tokens
    slot = tl.load(slot_mapping + current_token).to(tl.int64)
    if slot < 0:
        return

    physical_page = slot // PAGE_SIZE
    in_cache_token = slot % PAGE_SIZE
    key_row = tl.load(key + current_token * stride_key_token + offsets)
    key_cache_offset = (
        physical_page * stride_k_block
        + in_cache_token * stride_k_cache_token
        + offsets
    )
    tl.store(key_cache + key_cache_offset, key_row)
    tl.store(
        dense_key + logical_token * stride_dense_key_token + offsets,
        key_row,
    )

    value_row = tl.load(value + current_token * stride_value_token + offsets)
    value_cache_offset = (
        physical_page * stride_v_block
        + in_cache_token * stride_v_cache_token
        + offsets
    )
    tl.store(value_cache + value_cache_offset, value_row)
    tl.store(
        dense_value + logical_token * stride_dense_value_token + offsets,
        value_row,
    )


def gather_qwen35_paged_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    block_table: torch.Tensor,
    total_tokens: int,
) -> None:
    """Gather the exact Qwen3.5 784x4x256 NHD cache into dense K/V."""
    page_size = 784
    row_size = 4 * 256
    _gather_qwen35_nhd_pages_kernel[
        (page_size, triton.cdiv(total_tokens, page_size))
    ](
        key_cache,
        value_cache,
        dense_key,
        dense_value,
        block_table,
        total_tokens * row_size,
        key_cache.stride(0),
        value_cache.stride(0),
        PAGE_SIZE=page_size,
        ROW_SIZE=row_size,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )


def stage_qwen35_continuation_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    prefix_tokens: int,
    query_tokens: int,
    num_warps: int = 4,
) -> None:
    """Cache and stage one exact BF16 Qwen3.5 continuation chunk.

    Cached prefix rows are copied into the builder-owned dense workspace. The
    current chunk is loaded once from the post-RoPE K/V tensors and written to
    both its physical cache slots and the dense suffix. This preserves the
    existing dense FlashAttention input bit for bit while avoiding a second
    read of the just-written cache rows and one kernel launch.
    """
    page_size = 784
    num_kv_heads = 4
    head_size = 256
    row_size = num_kv_heads * head_size
    total_tokens = prefix_tokens + query_tokens

    assert prefix_tokens > 0 and query_tokens > 1
    assert num_warps in (4, 8)
    assert key.dtype == torch.bfloat16 and value.dtype == torch.bfloat16
    assert key_cache.dtype == torch.bfloat16
    assert value_cache.dtype == torch.bfloat16
    assert dense_key.dtype == torch.bfloat16
    assert dense_value.dtype == torch.bfloat16
    assert key.device == value.device == key_cache.device == value_cache.device
    assert dense_key.device == key.device and dense_value.device == key.device
    assert block_table.device == key.device and slot_mapping.device == key.device
    assert key.ndim == 3 and key.shape[1:] == (num_kv_heads, head_size)
    assert value.ndim == 3 and value.shape[1:] == (num_kv_heads, head_size)
    assert key.shape[0] >= query_tokens and value.shape[0] >= query_tokens
    assert key.stride(1) == head_size and key.stride(2) == 1
    assert value.stride(1) == head_size and value.stride(2) == 1
    assert key_cache.ndim == 4
    assert value_cache.ndim == 4
    assert key_cache.shape[1:] == (page_size, num_kv_heads, head_size)
    assert value_cache.shape[1:] == (page_size, num_kv_heads, head_size)
    assert key_cache.stride(2) == head_size and key_cache.stride(3) == 1
    assert value_cache.stride(2) == head_size and value_cache.stride(3) == 1
    assert dense_key.ndim == 3
    assert dense_value.ndim == 3
    assert dense_key.shape[0] >= total_tokens
    assert dense_value.shape[0] >= total_tokens
    assert dense_key.shape[1:] == (num_kv_heads, head_size)
    assert dense_value.shape[1:] == (num_kv_heads, head_size)
    assert dense_key.stride(1) == head_size and dense_key.stride(2) == 1
    assert dense_value.stride(1) == head_size and dense_value.stride(2) == 1
    assert block_table.ndim == 2 and block_table.shape[0] == 1
    assert block_table.shape[1] >= triton.cdiv(total_tokens, page_size)
    assert block_table.dtype == torch.int32
    assert block_table.stride(1) == 1
    assert slot_mapping.dtype == torch.int64
    assert slot_mapping.numel() >= query_tokens
    assert slot_mapping.stride(0) == 1

    _stage_qwen35_continuation_kv_kernel[
        (page_size, triton.cdiv(total_tokens, page_size))
    ](
        key,
        value,
        key_cache,
        value_cache,
        dense_key,
        dense_value,
        block_table,
        slot_mapping,
        prefix_tokens,
        total_tokens,
        key.stride(0),
        value.stride(0),
        key_cache.stride(0),
        value_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(1),
        dense_key.stride(0),
        dense_value.stride(0),
        PAGE_SIZE=page_size,
        ROW_SIZE=row_size,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=1,
    )
