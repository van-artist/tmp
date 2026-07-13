# Third-party notices

- vLLM / OpenDAS fork, Apache-2.0. Base commit `fa71803`.
- AMD AITER, MIT license: <https://github.com/ROCm/aiter>. This submission
  calls the AITER package already provided by the fixed environment; it does
  not copy AITER source or binaries.
- Triton, MIT license: <https://github.com/triton-lang/triton>. The modified
  prefix kernel is the existing vLLM/OpenDAS Triton source retained under the
  repository's original notices.
- FlashAttention, BSD-3-Clause license:
  <https://github.com/Dao-AILab/flash-attention>. This submission calls the
  ROCm FlashAttention package already bundled in the fixed environment after
  gathering the exact paged K/V values into a temporary dense buffer. It does
  not copy FlashAttention source or binaries.

Optimization ideas were also compared with participant-shared notes supplied
locally by the team. No text or source from those notes is included here.
