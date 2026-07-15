# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from .chunk import chunk_gated_delta_rule, chunk_gated_delta_rule_into
from .fused_recurrent import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from .fused_gdn_prefill_post_conv import (
    can_use_exact_local_cumsum,
    fused_post_conv_prep,
)
from .fused_gdn_rmsnorm import rmsnorm_gated_strided_z_inplace
from .fused_qwen35_prefill_conv import (
    can_use_qwen35_prefill_conv_fusion,
    fused_qwen35_prefill_conv_prep,
)
from .fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update
from .layernorm_guard import RMSNormGated

__all__ = [
    "RMSNormGated",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_into",
    "can_use_exact_local_cumsum",
    "can_use_qwen35_prefill_conv_fusion",
    "fused_post_conv_prep",
    "fused_qwen35_prefill_conv_prep",
    "rmsnorm_gated_strided_z_inplace",
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "fused_sigmoid_gating_delta_rule_update",
]
