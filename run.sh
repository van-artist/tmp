#!/bin/bash
set -e

MODEL_PATH="/public/home/xdzs2026_c284/Qwen3.5-27B"

echo "[Run] Starting vLLM baseline service..."
echo "[Run] Model path: ${MODEL_PATH}"

vllm serve "${MODEL_PATH}" \
  --max-model-len 32768
