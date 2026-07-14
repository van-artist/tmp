# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

from dataclasses import dataclass, replace

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    *,
    mamba_cache_mode: str = "none",
    cudagraph_mode: CUDAGraphMode | None = None,
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(block_size=BLOCK_SIZE)
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    if cudagraph_mode is not None:
        vllm_config.compilation_config.cudagraph_mode = cudagraph_mode
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode=mamba_cache_mode,
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


@pytest.mark.parametrize(
    "batch",
    [
        BatchSpec(seq_lens=[65, 33], query_lens=[1, 1]),
        BatchSpec(seq_lens=[65, 33], query_lens=[1, 9]),
        BatchSpec(seq_lens=[65, 33], query_lens=[9, 17]),
    ],
    ids=["decode", "mixed_decode_prefill", "prefill"],
)
def test_update_block_table_reuses_metadata_without_mutating_source(
    batch: BatchSpec,
):
    builder_a = _create_gdn_builder()
    builder_b = _create_gdn_builder()
    builder_c = _create_gdn_builder()
    common_a = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common_b = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common_c = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common_a.block_table_tensor.fill_(11)
    common_b.block_table_tensor.fill_(22)
    common_c.block_table_tensor.fill_(33)

    meta_a = builder_a.build(0, common_a)
    assert meta_a.non_spec_state_indices_tensor is not None
    assert meta_a.non_spec_query_start_loc is not None
    source_state_indices = meta_a.non_spec_state_indices_tensor.clone()
    source_query_start_loc = meta_a.non_spec_query_start_loc.clone()
    meta_b = builder_b.update_block_table(
        meta_a, common_b.block_table_tensor, common_b.slot_mapping
    )
    meta_c = builder_c.update_block_table(
        meta_a, common_c.block_table_tensor, common_c.slot_mapping
    )

    assert builder_a.supports_update_block_table
    assert meta_b is not meta_a
    assert meta_c is not meta_a
    torch.testing.assert_close(
        meta_b.non_spec_state_indices_tensor,
        torch.full_like(meta_b.non_spec_state_indices_tensor, 22),
    )
    torch.testing.assert_close(
        meta_c.non_spec_state_indices_tensor,
        torch.full_like(meta_c.non_spec_state_indices_tensor, 33),
    )
    torch.testing.assert_close(
        meta_a.non_spec_state_indices_tensor, source_state_indices
    )
    torch.testing.assert_close(meta_a.non_spec_query_start_loc, source_query_start_loc)
    assert meta_b.num_decodes == meta_a.num_decodes
    assert meta_b.num_prefills == meta_a.num_prefills
    assert meta_b.has_initial_state is meta_a.has_initial_state
    assert meta_b.non_spec_query_start_loc is meta_a.non_spec_query_start_loc


def test_regular_decode_skips_context_length_computation(monkeypatch):
    builder = _create_gdn_builder()
    batch = BatchSpec(seq_lens=[65], query_lens=[1])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)

    def fail_if_called(_self):
        raise AssertionError("decode should not compute prefill context lengths")

    monkeypatch.setattr(
        type(common), "compute_num_computed_tokens", fail_if_called
    )
    metadata = builder.build(0, common)

    assert metadata.num_decodes == 1
    assert metadata.num_prefills == 0


def test_update_block_table_uses_current_cudagraph_buffers():
    builder_a = _create_gdn_builder(cudagraph_mode=CUDAGraphMode.FULL)
    builder_b = _create_gdn_builder(cudagraph_mode=CUDAGraphMode.FULL)

    batch = BatchSpec(seq_lens=[65], query_lens=[1])
    common_a = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common_b = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    common_a.block_table_tensor.fill_(11)
    common_b.block_table_tensor.fill_(22)

    meta_a = builder_a.build(0, common_a)
    assert meta_a.non_spec_state_indices_tensor is not None
    assert meta_a.non_spec_query_start_loc is not None
    source_state_indices = meta_a.non_spec_state_indices_tensor.clone()
    source_query_start_loc = meta_a.non_spec_query_start_loc.clone()
    padded_meta_a = replace(meta_a, num_actual_tokens=3)
    actual = builder_b.update_block_table(
        padded_meta_a, common_b.block_table_tensor, common_b.slot_mapping
    )

    def shares_storage(tensor: torch.Tensor, buffer: torch.Tensor) -> bool:
        return (
            tensor.untyped_storage().data_ptr()
            == buffer.untyped_storage().data_ptr()
        )

    assert actual.non_spec_state_indices_tensor is not None
    assert actual.non_spec_query_start_loc is not None
    assert shares_storage(
        actual.non_spec_state_indices_tensor,
        builder_b.non_spec_state_indices_tensor,
    )
    assert shares_storage(
        actual.non_spec_query_start_loc,
        builder_b.non_spec_query_start_loc,
    )
    assert not shares_storage(
        actual.non_spec_state_indices_tensor,
        builder_a.non_spec_state_indices_tensor,
    )
    torch.testing.assert_close(
        actual.non_spec_state_indices_tensor,
        torch.tensor([22, PAD_SLOT_ID, PAD_SLOT_ID], dtype=torch.int32),
    )
    torch.testing.assert_close(
        actual.non_spec_query_start_loc,
        torch.tensor([0, 1, 1, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        meta_a.non_spec_state_indices_tensor, source_state_indices
    )
    torch.testing.assert_close(meta_a.non_spec_query_start_loc, source_query_start_loc)


def test_update_block_table_rejects_short_block_table():
    builder = _create_gdn_builder()
    batch = BatchSpec(seq_lens=[65, 33], query_lens=[1, 1])
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    metadata = builder.build(0, common)
    short_block_table = common.block_table_tensor[:1]

    with pytest.raises(AssertionError):
        builder.update_block_table(metadata, short_block_table, common.slot_mapping)


def test_update_block_table_disabled_for_spec_decode():
    builder = _create_gdn_builder(num_speculative_tokens=2)
    assert not builder.supports_update_block_table


@pytest.mark.parametrize("mamba_cache_mode", ["align", "all"])
def test_update_block_table_disabled_for_other_cache_modes(mamba_cache_mode: str):
    builder = _create_gdn_builder(mamba_cache_mode=mamba_cache_mode)
    assert not builder.supports_update_block_table
