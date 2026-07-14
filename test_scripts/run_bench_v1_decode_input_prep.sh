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

INSTALLED_ROOT="$(
    cd /tmp
    PYTHONPATH="" python - <<'PY'
from pathlib import Path

import vllm

print(Path(vllm.__file__).resolve().parent)
PY
)"
INSTALLED_ROOT="${INSTALLED_ROOT##*$'\n'}"
if [[ ! -d "$INSTALLED_ROOT" ]]; then
    echo "Could not resolve installed vLLM root: $INSTALLED_ROOT" >&2
    exit 1
fi
for relative_file in \
    v1/worker/gpu_decode_input_prep.py \
    v1/worker/gpu_model_runner.py \
    v1/worker/block_table.py; do
    source_file="$PROJECT_ROOT/vllm/$relative_file"
    installed_file="$INSTALLED_ROOT/$relative_file"
    if [[ ! -f "$installed_file" ]]; then
        echo "Missing installed source: $installed_file" >&2
        exit 1
    fi
    source_sha="$(sha256sum "$source_file" | cut -d' ' -f1)"
    installed_sha="$(sha256sum "$installed_file" | cut -d' ' -f1)"
    if [[ "$source_sha" != "$installed_sha" ]]; then
        echo "Installed source does not match: $relative_file" >&2
        exit 1
    fi
done

echo "=== Running V1 single-token input-prep A/B/A benchmark ==="
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Imported root: $INSTALLED_ROOT"
has_output=0
for arg in "$@"; do
    if [[ "$arg" == "--output" || "$arg" == --output=* ]]; then
        has_output=1
        break
    fi
done
if [[ "$has_output" == 0 ]]; then
    set -- "$@" --output "$SCRIPT_DIR/v1_decode_input_prep.csv"
fi
cd /tmp
unset PYTHONPATH
exec python "$SCRIPT_DIR/bench_v1_decode_input_prep.py" "$@"
