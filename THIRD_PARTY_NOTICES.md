# Third-party notices

- vLLM / OpenDAS fork, Apache-2.0. Base commit `fa71803`.
- AMD AITER, MIT license: <https://github.com/ROCm/aiter>. This submission
  calls the AITER package already provided by the fixed environment. The
  MFMA tiling and launch configuration in
  `vllm/model_executor/layers/qwen35_gateup_silu.py` is adapted from AITER's
  `aiter/ops/triton/gemm_a16w16.py`; the dual accumulation,
  submission-specific shape guard, BF16 rounding sequence and SwiGLU output
  are local modifications. No AITER binary is copied.
- Triton, MIT license: <https://github.com/triton-lang/triton>. The modified
  prefix kernel is the existing vLLM/OpenDAS Triton source retained under the
  repository's original notices.
- Flash Linear Attention, MIT license:
  <https://github.com/fla-org/flash-linear-attention>. The FLA-derived files
  already present in the OpenDAS/vLLM base retain their original Songlin Yang,
  Yu Zhang and Tri Dao copyright/license headers. The GDN prefill fusion
  backported by upstream vLLM commit `4eefbf9609e5` retains the vLLM
  Apache-2.0 notice.
- FlashAttention, BSD-3-Clause license:
  <https://github.com/Dao-AILab/flash-attention>. This submission calls the
  ROCm FlashAttention package already bundled in the fixed environment after
  gathering the exact paged K/V values into a temporary dense buffer. It does
  not copy FlashAttention source or binaries.

Optimization ideas were also compared with participant-shared notes supplied
locally by the team. No text or source from those notes is included here.
