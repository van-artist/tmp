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
