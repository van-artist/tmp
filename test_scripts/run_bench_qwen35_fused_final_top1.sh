#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/env_setup.sh"

echo "=== Building vLLM from the current worktree ==="
python setup.py bdist_wheel

WHEEL_PATH=""
for candidate in "$PROJECT_ROOT"/dist/vllm-*.whl; do
    [[ -e "$candidate" ]] || continue
    if [[ -z "$WHEEL_PATH" || "$candidate" -nt "$WHEEL_PATH" ]]; then
        WHEEL_PATH="$candidate"
    fi
done
if [[ -z "$WHEEL_PATH" ]]; then
    echo "No vLLM wheel found under $PROJECT_ROOT/dist" >&2
    exit 1
fi

echo "=== Installing $WHEEL_PATH ==="
python -m pip install "$WHEEL_PATH" --no-deps --force-reinstall

echo "=== Verifying installed ROCm extension ==="
cd /tmp
PYTHONPATH="" python - <<'PY'
from importlib import import_module
from pathlib import Path

import torch
import vllm

extension = import_module("vllm._rocm_C")
required = ("LLMM1", "LLMM1Top1", "LLMM1RMSNormTop1")
missing = [name for name in required if not hasattr(torch.ops._rocm_C, name)]
if missing:
    raise RuntimeError(f"Installed vLLM is missing ROCm operators: {missing}")

print(f"Imported vLLM: {Path(vllm.__file__).resolve()}")
print(f"Imported ROCm extension: {Path(extension.__file__).resolve()}")
print("Registered ops: " + ", ".join(f"_rocm_C::{name}" for name in required))
PY

echo "=== Running Qwen3.5 fused final-tail A/B/A benchmark ==="
echo "Git commit: $(git -C "$PROJECT_ROOT" rev-parse --short HEAD)"
unset PYTHONPATH
exec python "$SCRIPT_DIR/bench_qwen35_fused_final_top1.py" "$@"
