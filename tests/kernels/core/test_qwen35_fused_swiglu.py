# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_swiglu import (  # noqa: F401
    qwen35_fused_gate_up_swiglu_impl,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Qwen3.5 fused SwiGLU/down GEMV is a gfx936 ROCm specialization",
)
@torch.inference_mode()
def test_qwen35_fused_gate_up_swiglu_matches_reference():
    from vllm.platforms.rocm import on_gfx936

    if not on_gfx936():
        pytest.skip("exact kernel is enabled only on gfx936")

    set_random_seed(13)
    x = torch.randn((1, 5120), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        (34816, 5120), device="cuda", dtype=torch.bfloat16
    )

    gate_up = F.linear(x, weight)
    gate, up = gate_up.chunk(2, dim=-1)
    reference = F.silu(gate) * up
    actual = torch.ops.vllm.qwen35_fused_gate_up_swiglu(x, weight)

    torch.testing.assert_close(actual, reference, atol=0.125, rtol=0.02)
