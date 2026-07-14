# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.logits_processor.state import LogitsProcessors


class _ConservativeCustomProcessor(LogitsProcessor):
    def __init__(self, *_):
        pass

    def apply(self, logits):
        return logits

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update) -> None:
        pass


def test_idle_builtin_non_argmax_processors_are_inactive():
    logit_bias = object.__new__(LogitBiasLogitsProcessor)
    logit_bias.biases = {}
    min_tokens = object.__new__(MinTokensLogitsProcessor)
    min_tokens.min_toks = {}
    processors = LogitsProcessors([logit_bias, min_tokens])

    assert not processors.has_active_non_argmax_invariant

    logit_bias.biases[0] = {1: 2.0}
    assert processors.has_active_non_argmax_invariant

    logit_bias.biases.clear()
    min_tokens.min_toks[0] = (1, [], {2})
    assert processors.has_active_non_argmax_invariant


def test_custom_non_argmax_processor_is_active_by_default():
    processors = LogitsProcessors([_ConservativeCustomProcessor()])

    assert processors.has_active_non_argmax_invariant
