#!/usr/bin/env bash
# Evidence-poisoning experiment: 27 cases x 2 conditions.
set -euo pipefail
cd "$(dirname "$0")/.."
export RAG_CONFIG="${RAG_CONFIG:-configs/injection.yaml}"
python src/run_injection.py
