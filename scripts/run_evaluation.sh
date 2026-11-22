#!/usr/bin/env bash
# Recompute every metric and figure from the committed predictions (no API key needed).
set -euo pipefail
cd "$(dirname "$0")/.."
python src/preprocessing.py
python src/evaluate_metrics.py
python src/evaluate_hallucination.py
python src/filter_baseline.py
python src/make_figure1.py
python src/plot_ablation.py
