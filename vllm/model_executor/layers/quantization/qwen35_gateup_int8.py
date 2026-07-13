# SPDX-License-Identifier: Apache-2.0
"""gfx936-only runtime INT8 path for the dense Qwen3.5 gate/up GEMM.

This is intentionally not a general-purpose quantization method.  The BW200
torch build has a fast ``aten::_int_mm`` for the exact decode shape, but no
usable native FP8 linear kernel.  Two INT8 copies replace the original BF16
weight without increasing per-layer weight storage:

* [34816, 5120], row major, for inexpensive prefill dequantization; and
* [5120, 34816], contiguous, for ``torch._int_mm`` during one-token decode.

All selected layers share one BF16 prefill scratch tensor.  Decode uses a
17-row input pad because this ROCm ``_int_mm`` implementation requires M > 16.
The custom ops explicitly describe their mutations so Dynamo/AOT and HIP graph
capture preserve scratch/input-pad dependencies between adjacent layers.
"""

import re
from typing import Any

import torch
import torch.nn.functional as F

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_QWEN35_GATEUP_N = 34816
_QWEN35_GATEUP_K = 5120
_QWEN35_NUM_LAYERS = 64
_QWEN35_INTMM_MPAD = 17

# A single engine/model is created in the contest service.  Keeping this map
# process-local also prevents one 340 MiB scratch allocation per linear layer.
_SHARED_SCRATCH: dict[tuple[int, torch.dtype, int, int], torch.Tensor] = {}
_INT_MM_PROBED: set[int] = set()
_LOGGED_MODES: set[str] = set()


def _device_index(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    return torch.cuda.current_device() if index is None else index


def _on_exact_gfx936(device_index: int | None = None) -> bool:
    if not current_platform.is_rocm() or not torch.cuda.is_available():
        return False
    index = torch.cuda.current_device() if device_index is None else device_index
    arch = str(torch.cuda.get_device_properties(index).gcnArchName)
    return arch.split(":", 1)[0] == "gfx936"


def _selected_layer(layer_idx: int) -> bool:
    mode = str(envs.VLLM_ROCM_QWEN35_GATEUP_INT8_LAYERS).strip().lower()
    if mode in ("", "0", "off", "false", "none"):
        return False
    if mode in ("1", "64", "all"):
        return True
    # The default 32-layer policy keeps the first and last 16 layers in BF16.
    if mode in ("32", "middle32", "mid32"):
        return 16 <= layer_idx < 48
    if mode in ("even32", "even"):
        return layer_idx % 2 == 0
    if mode in ("last32", "late32"):
        return layer_idx >= 32
    raise ValueError(
        "VLLM_ROCM_QWEN35_GATEUP_INT8_LAYERS must be one of "
        "0, 32/middle32, even32, last32, or 64/all; got " + repr(mode)
    )


def _strict_candidate_guard(layer: torch.nn.Module) -> tuple[bool, int]:
    """Return (is_selected_exact_Qwen_layer, layer_idx)."""
    if not getattr(layer, "_qwen35_dense_gateup_candidate", False):
        return False, -1
    if not current_platform.is_rocm() or not _on_exact_gfx936():
        return False, -1
    if getattr(layer, "tp_size", None) != 1:
        return False, -1
    if getattr(layer, "_qwen35_num_hidden_layers", None) != _QWEN35_NUM_LAYERS:
        return False, -1

    layer_idx = int(getattr(layer, "_qwen35_layer_idx", -1))
    prefix = str(getattr(layer, "prefix", ""))
    match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp\.gate_up_proj$", prefix)
    if match is None or int(match.group(1)) != layer_idx:
        return False, -1
    if not 0 <= layer_idx < _QWEN35_NUM_LAYERS:
        return False, -1
    if not _selected_layer(layer_idx):
        return False, layer_idx

    # Runtime import avoids a circular import while linear.py is initialized.
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    if not isinstance(layer, MergedColumnParallelLinear):
        return False, -1
    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return False, -1
    if (
        weight.shape != (_QWEN35_GATEUP_N, _QWEN35_GATEUP_K)
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
    ):
        return False, -1
    return True, layer_idx


@triton.jit
def _quantize_weight_rows_kernel(
    weight_ptr,
    qweight_ptr,
    scale_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    values = tl.load(
        weight_ptr + row * K + cols, mask=mask, other=0.0
    ).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-10)
    scale = absmax / 127.0
    quantized = tl.extra.hip.libdevice.round(values / scale)
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0).to(tl.int8)
    tl.store(qweight_ptr + row * K + cols, quantized, mask=mask)
    tl.store(scale_ptr + row, scale)


def _quantize_weight_rows(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    qweight = torch.empty_like(weight, dtype=torch.int8)
    scale = torch.empty(
        (weight.shape[0],), device=weight.device, dtype=torch.float32
    )
    _quantize_weight_rows_kernel[(weight.shape[0],)](
        weight,
        qweight,
        scale,
        N=weight.shape[0],
        K=weight.shape[1],
        BLOCK_K=8192,
        num_warps=8,
    )
    return qweight, scale


@triton.jit
def _quantize_decode_input_kernel(
    x_ptr,
    qpad_ptr,
    xscale_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    values = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-10)
    scale = absmax / 127.0
    quantized = tl.extra.hip.libdevice.round(values / scale)
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0).to(tl.int8)
    # Rows 1:17 are initialized to zero once after loading and never written.
    tl.store(qpad_ptr + cols, quantized, mask=mask)
    tl.store(xscale_ptr, scale)


def _qwen35_gateup_quant_input(
    x: torch.Tensor,
    qpad: torch.Tensor,
    xscale: torch.Tensor,
) -> None:
    _quantize_decode_input_kernel[(1,)](
        x,
        qpad,
        xscale,
        K=_QWEN35_GATEUP_K,
        BLOCK_K=8192,
        num_warps=16,
    )


def _qwen35_gateup_quant_input_fake(
    x: torch.Tensor,
    qpad: torch.Tensor,
    xscale: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen35_gateup_quant_input",
    op_func=_qwen35_gateup_quant_input,
    mutates_args=["qpad", "xscale"],
    fake_impl=_qwen35_gateup_quant_input_fake,
)


@triton.jit
def _dequantize_decode_output_kernel(
    accum_ptr,
    xscale_ptr,
    wscale_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    accum = tl.load(accum_ptr + cols, mask=mask, other=0).to(tl.float32)
    scale = tl.load(xscale_ptr) * tl.load(
        wscale_ptr + cols, mask=mask, other=0.0
    )
    tl.store(output_ptr + cols, accum * scale, mask=mask)


def _qwen35_gateup_dequant_output(
    accum: torch.Tensor,
    xscale: torch.Tensor,
    wscale: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        (1, wscale.shape[0]), device=accum.device, dtype=torch.bfloat16
    )
    _dequantize_decode_output_kernel[
        (triton.cdiv(wscale.shape[0], 1024),)
    ](
        accum,
        xscale,
        wscale,
        output,
        N=wscale.shape[0],
        BLOCK_N=1024,
        num_warps=8,
    )
    return output


def _qwen35_gateup_dequant_output_fake(
    accum: torch.Tensor,
    xscale: torch.Tensor,
    wscale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (1, wscale.shape[0]), device=accum.device, dtype=torch.bfloat16
    )


direct_register_custom_op(
    op_name="qwen35_gateup_dequant_output",
    op_func=_qwen35_gateup_dequant_output,
    mutates_args=[],
    fake_impl=_qwen35_gateup_dequant_output_fake,
)


@triton.jit
def _dequantize_weight_kernel(
    qweight_ptr,
    scale_ptr,
    scratch_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N * K
    rows = offsets // K
    values = tl.load(qweight_ptr + offsets, mask=mask, other=0).to(tl.float32)
    scales = tl.load(scale_ptr + rows, mask=mask, other=0.0)
    tl.store(scratch_ptr + offsets, values * scales, mask=mask)


def _qwen35_gateup_dequant_weight(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    scratch: torch.Tensor,
) -> None:
    n, k = qweight.shape
    _dequantize_weight_kernel[(triton.cdiv(n * k, 2048),)](
        qweight,
        scale,
        scratch,
        N=n,
        K=k,
        BLOCK=2048,
        num_warps=4,
    )


def _qwen35_gateup_dequant_weight_fake(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    scratch: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen35_gateup_dequant_weight",
    op_func=_qwen35_gateup_dequant_weight,
    mutates_args=["scratch"],
    fake_impl=_qwen35_gateup_dequant_weight_fake,
)


def _shared_scratch(weight: torch.Tensor) -> torch.Tensor:
    key = (
        _device_index(weight),
        weight.dtype,
        weight.shape[0],
        weight.shape[1],
    )
    scratch = _SHARED_SCRATCH.get(key)
    if scratch is None:
        scratch = torch.empty_like(weight)
        _SHARED_SCRATCH[key] = scratch
    return scratch


def maybe_process_qwen35_gateup_int8(layer: torch.nn.Module) -> bool:
    """Quantize an exact target layer after checkpoint loading.

    Returns True only when the BF16 parameter was successfully replaced.
    No tensor on an unselected/unsupported layer is modified.
    """
    selected, layer_idx = _strict_candidate_guard(layer)
    if not selected:
        return False

    weight = layer.weight.detach()
    qweight_row, weight_scale = _quantize_weight_rows(weight)
    # This second, contiguous orientation is what makes _int_mm fast on gfx936.
    qweight_kn = qweight_row.t().contiguous()
    input_pad = torch.zeros(
        (_QWEN35_INTMM_MPAD, _QWEN35_GATEUP_K),
        device=weight.device,
        dtype=torch.int8,
    )
    input_scale = torch.empty((), device=weight.device, dtype=torch.float32)
    scratch = _shared_scratch(weight)

    # Fail before releasing BF16 if the installed torch/ROCm build cannot run
    # this exact aten kernel.  One successful probe covers all identical layers.
    device_idx = _device_index(weight)
    if device_idx not in _INT_MM_PROBED:
        probe = torch._int_mm(input_pad, qweight_kn)
        torch.cuda.synchronize(weight.device)
        del probe
        _INT_MM_PROBED.add(device_idx)

    layer.register_buffer(
        "qwen35_gateup_qweight_row", qweight_row, persistent=False
    )
    layer.register_buffer(
        "qwen35_gateup_qweight_kn", qweight_kn, persistent=False
    )
    layer.register_buffer(
        "qwen35_gateup_weight_scale", weight_scale, persistent=False
    )
    layer.register_buffer(
        "qwen35_gateup_input_pad", input_pad, persistent=False
    )
    layer.register_buffer(
        "qwen35_gateup_input_scale", input_scale, persistent=False
    )
    # Registering the same Tensor on every selected layer does not duplicate its
    # storage.  As a buffer it is lifted explicitly by torch.compile.
    layer.register_buffer(
        "qwen35_gateup_shared_scratch", scratch, persistent=False
    )

    # Only publish the dispatch flag after every allocation/probe succeeds.
    layer._qwen35_gateup_int8_enabled = True
    del layer._parameters["weight"]
    del weight

    mode = str(envs.VLLM_ROCM_QWEN35_GATEUP_INT8_LAYERS).strip().lower()
    if mode not in _LOGGED_MODES:
        logger.info(
            "Enabled gfx936 Qwen3.5 gate_up INT8 mode=%s (first layer=%d)",
            mode,
            layer_idx,
        )
        _LOGGED_MODES.add(mode)
    return True


def apply_qwen35_gateup_int8(
    layer: Any,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Apply the post-load Qwen gate/up representation."""
    if bias is not None:
        raise RuntimeError("Qwen3.5 gate_up INT8 path requires bias=None")
    if x.shape[-1] != _QWEN35_GATEUP_K or x.dtype != torch.bfloat16:
        raise RuntimeError(
            "Qwen3.5 gate_up INT8 received an unexpected input shape/dtype"
        )

    x_2d = x.reshape(-1, _QWEN35_GATEUP_K)
    if x_2d.shape[0] == 1:
        torch.ops.vllm.qwen35_gateup_quant_input(
            x_2d,
            layer.qwen35_gateup_input_pad,
            layer.qwen35_gateup_input_scale,
        )
        accum = torch._int_mm(
            layer.qwen35_gateup_input_pad,
            layer.qwen35_gateup_qweight_kn,
        )
        output = torch.ops.vllm.qwen35_gateup_dequant_output(
            accum,
            layer.qwen35_gateup_input_scale,
            layer.qwen35_gateup_weight_scale,
        )
    else:
        # Mutation is explicit in the custom-op schema.  The following linear
        # read therefore cannot be reordered after the next layer's overwrite.
        torch.ops.vllm.qwen35_gateup_dequant_weight(
            layer.qwen35_gateup_qweight_row,
            layer.qwen35_gateup_weight_scale,
            layer.qwen35_gateup_shared_scratch,
        )
        output = F.linear(x_2d, layer.qwen35_gateup_shared_scratch)

    return output.reshape(*x.shape[:-1], _QWEN35_GATEUP_N)
