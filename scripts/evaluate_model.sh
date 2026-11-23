#!/bin/bash
# Usage: ./scripts/evaluate_model.sh MODEL_NAME [PERSPECTIVE_API_KEY]
# Run from the repository root. Set PYTHON to choose the interpreter (default: python).
set -euo pipefail

MODEL_NAME=${1:?usage: $0 MODEL_NAME [PERSPECTIVE_API_KEY]}
API_KEY=${2:-}
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

echo "Running evaluation for: $MODEL_NAME"
if [ -n "$API_KEY" ]; then
  "$PYTHON" scripts/run_all.py --model "$MODEL_NAME" --key "$API_KEY"
else
  "$PYTHON" scripts/run_all.py --model "$MODEL_NAME"
fi
