# AI Use Disclosure

Course 5801 final project — "A RAG + Multi-Agent Question-Answering System for
Big-Data Laboratory Troubleshooting". Author: Yajie Gao.

## 1. What the AI assistant did

| Area | AI involvement | Human verification |
|---|---|---|
| Experiment code (`pipeline.py`, `run_*.py`, `filter_baseline.py`) | Drafted and refactored together with the author during the semester | Author ran every script, inspected the API logs, and fixed the failures (empty outputs, forced thinking budget, retry ladder) |
| Evaluation code (`evaluate_metrics.py`, `evaluate_hallucination.py`) | Drafted by the AI assistant from the definitions agreed with the author | Every metric was re-derived from the committed prediction files and compared against the numbers in the paper; all match to four decimals |
| Figures (`make_figure1.py`, `plot_ablation.py`) | Drafted by the AI assistant, layout iterated with the author | Author specified the values and checked each rendered figure against Table 3 |
| Manuscript | The assistant proposed rewrites of sections 3–6, checked reference metadata, and produced the Chinese→paper formatting | Author decided on every claim, rewrote the argument where needed, and verified each reference by opening it |
| English text in this repository (README, docstrings) | Drafted with AI assistance | Author reviewed the content for accuracy |

## 2. What the AI assistant did **not** do

* It did **not** run the experiments. All 977 model calls (700 for the main experiment,
  169 for the bare baseline including retries, 108 for the injection experiment: 27 cases ×
  [1 single-pass RAG call + 3 pipeline calls]) were launched
  by the author on the author's machine; the timings and token counts in `results/` are
  measured, not estimated. The per-run breakdown is in `results/experiment_log.csv`.
* It did **not** fill in the human evaluation sheets. The AI drafted a suggested score for
  each of the 120 judgements (20 cases × 3 dimensions × baseline/pipeline: VC 40, FS 40, ER 40); the author read every prediction
  against the evidence and adopted all of the suggestions (agreement 100%), so the shipped
  sheets (`results/fs_er_human_scores_20.csv`, `results/vc_manual_scores_20.csv`) coincide
  with the draft and only one version is committed. The coding rules are printed in
  `results/vc_scoring_template_20.csv`. Because the sheets come from a single annotator with AI
  pre-annotation, the paper describes them as a consistency check, **not** as an independent
  blind human evaluation.
* It did **not** invent data. An earlier draft of the paper contained placeholder numbers
  produced by a rule-based simulation; those were detected, discarded, and replaced by the
  real API runs committed here. The provenance of every current number is listed in
  `results/experiment_log.csv` and `docs/results_summary_zh.md`.

## 3. Third-party code

No external source code was copied. The libraries used (`openai`, `numpy`, `matplotlib`,
`scipy`, `PyYAML`) are consumed as dependencies through their public APIs.
The agent prompts are original; the papers that inspired them are cited in the manuscript.

## 4. Statement

The author understands the full content of this repository and can explain and re-run every
script. The numbers in the final paper correspond to the files in `results/`, and those files
are reproduced by `src/evaluate_metrics.py` from the committed per-case predictions.
