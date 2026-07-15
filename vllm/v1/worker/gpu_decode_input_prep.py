# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.block_table import BlockTable


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)


@triton.jit
def _prepare_single_token_decode_kernel(
    sampled_token_ids_ptr,
    input_ids_ptr,
    positions_ptr,
    mrope_positions_ptr,
    mrope_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    discard_request_mask_ptr,
    is_token_ids_ptr,
    block_table_ptrs,
    block_sizes,
    slot_mapping_ptrs,
    position,
    mrope_delta,
    MAX_NUM_REQS: tl.constexpr,
    META_BLOCK_SIZE: tl.constexpr,
    USE_MROPE: tl.constexpr,
    WRITE_IS_TOKEN_IDS: tl.constexpr,
):
    group_id = tl.program_id(0)

    if group_id == 0:
        token_id = tl.load(sampled_token_ids_ptr)
        tl.store(input_ids_ptr, token_id)

        if USE_MROPE:
            mrope_offsets = tl.arange(0, 4)
            tl.store(
                mrope_positions_ptr + mrope_offsets * mrope_stride,
                position + mrope_delta,
                mask=mrope_offsets < 3,
            )
        else:
            tl.store(positions_ptr, position)

        if WRITE_IS_TOKEN_IDS:
            tl.store(is_token_ids_ptr, 1)

        offsets = tl.arange(0, META_BLOCK_SIZE)
        tl.store(
            query_start_loc_ptr + offsets,
            tl.where(offsets == 0, 0, 1),
            mask=offsets < MAX_NUM_REQS + 1,
        )
        tl.store(
            seq_lens_ptr + offsets,
            tl.where(offsets == 0, position + 1, 0),
            mask=offsets < MAX_NUM_REQS,
        )
        tl.store(
            discard_request_mask_ptr + offsets,
            0,
            mask=offsets < MAX_NUM_REQS,
        )

    block_size = tl.load(block_sizes + group_id)
    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_number = tl.load(
        block_table_ptr + position // block_size,
    )
    slot = block_number * block_size + position % block_size
    slot_mapping_ptr = _load_ptr(slot_mapping_ptrs + group_id, tl.int64)
    tl.store(slot_mapping_ptr, slot)


class V1SingleTokenDecodeInputPreparer:
    """GPU-resident V1 input preparation for a single decode token.

    The caller is responsible for restricting this path to one request, one
    scheduled token, and context-parallel size one. The block-table and output
    buffers are persistent, so only the sampled token tensor and scalar decode
    position change between launches.
    """

    def __init__(
        self,
        block_tables: Sequence[BlockTable],
        max_num_reqs: int,
        device: torch.device,
    ) -> None:
        if not block_tables:
            raise ValueError("At least one block table is required")
        if any(
            table.pcp_world_size != 1 or table.dcp_world_size != 1
            for table in block_tables
        ):
            raise ValueError("Context parallelism is not supported")

        self.num_groups = len(block_tables)
        self.max_num_reqs = max_num_reqs
        self.meta_block_size = triton.next_power_of_2(max_num_reqs + 1)
        self.block_table_ptrs = torch.tensor(
            [table.block_table.gpu.data_ptr() for table in block_tables],
            dtype=torch.uint64,
            device=device,
        )
        self.block_sizes = torch.tensor(
            [table.block_size for table in block_tables],
            dtype=torch.int32,
            device=device,
        )
        self.slot_mapping_ptrs = torch.tensor(
            [table.slot_mapping.gpu.data_ptr() for table in block_tables],
            dtype=torch.uint64,
            device=device,
        )
        self.logits_indices = torch.zeros(1, dtype=torch.int64, device=device)

    def prepare(
        self,
        sampled_token_ids: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mrope_positions: torch.Tensor | None,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        discard_request_mask: torch.Tensor,
        is_token_ids: torch.Tensor,
        position: int,
        mrope_delta: int,
        write_is_token_ids: bool,
    ) -> torch.Tensor:
        mrope_output = mrope_positions if mrope_positions is not None else positions
        mrope_stride = mrope_output.stride(0) if mrope_positions is not None else 0
        _prepare_single_token_decode_kernel[(self.num_groups,)](
            sampled_token_ids,
            input_ids,
            positions,
            mrope_output,
            mrope_stride,
            query_start_loc,
            seq_lens,
            discard_request_mask,
            is_token_ids,
            self.block_table_ptrs,
            self.block_sizes,
            self.slot_mapping_ptrs,
            position,
            mrope_delta,
            MAX_NUM_REQS=self.max_num_reqs,
            META_BLOCK_SIZE=self.meta_block_size,
            USE_MROPE=mrope_positions is not None,
            WRITE_IS_TOKEN_IDS=write_is_token_ids,
        )
        return self.logits_indices
