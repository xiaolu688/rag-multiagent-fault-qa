#!/usr/bin/env bash
# Main experiment: 100 test cases x 4 configurations (700 model calls).
set -euo pipefail
cd "$(dirname "$0")/.."
export RAG_CONFIG="${RAG_CONFIG:-configs/proposed_model.yaml}"
python src/run_real_experiment.py --limit "${LIMIT:-5}"   # smoke test first
if [ "${LIMIT:-5}" -lt 100 ]; then
  python src/run_real_experiment.py
fi
python src/run_bare_baseline.py
