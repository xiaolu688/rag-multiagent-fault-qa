# -*- coding: utf-8 -*-
"""Recompute every metric reported in the paper from per-case predictions.

Inputs
    results/case_comparison_100.csv   predictions of baseline / proposer /
                                      reviser / full pipeline for 100 cases
    results/bare_preds.csv            bare-LLM predictions (no evidence)
    data/bigdata_error_cases_500_en.csv   evidence span + gold answer per case

Output
    results/metrics_recomputed.csv    (should equal results/metrics_real.csv)

Metric definitions (all documented in the paper, Appendix C.5 / Sec. 4.3)
    strict EM        exact string equality after whitespace normalisation
    word F1          token sets T(x) = re.findall(r"[a-z0-9]+", lower(x));
                     F1 = 2|A∩B| / (|A|+|B|)            (sets, not bags)
    ROUGE-L          longest common subsequence over the same token stream,
                     reported as 2L/(m+n)
    version halluc.  a version number in the answer that is absent from
                     evidence ∪ gold counts as one hallucinated case
    refusal          answer contains "NOT COVERED" (agent declines)
    three sections   [Root cause] / [Version ...] / [Repair ...] all present
    length           token count with the same [a-z0-9]+ tokenizer
    McNemar          exact two-sided binomial test on discordant pairs
    Wilson CI        95% interval for a proportion

Usage
    python src/evaluate_metrics.py
    python src/evaluate_metrics.py --pred results/case_comparison_100.csv
"""
import argparse
import csv
import math
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOKEN = re.compile(r"[a-z0-9]+")
VER = re.compile(r"\b\d+\.\d+(?:\.\d+){0,2}\b")
CONFIGS = ["baseline", "proposer", "reviser", "pipeline"]
LABEL = {"baseline": "Single RAG (oracle ctx)", "proposer": "Proposer only",
         "reviser": "Proposer + Reviser", "pipeline": "Full multi-agent"}


def tokens(text):
    return TOKEN.findall(str(text or "").lower())


def strict_em(pred, gold):
    norm = lambda s: re.sub(r"\s+", " ", str(s or "")).strip().lower()
    return 1 if norm(pred) == norm(gold) else 0


def word_f1(pred, gold):
    a, b = set(tokens(pred)), set(tokens(gold))
    if not a or not b:
        return 0.0
    c = len(a & b)
    return 2 * c / (len(a) + len(b)) if c else 0.0


def lcs_len(a, b):
    n = len(b)
    prev = [0] * (n + 1)
    for i in range(len(a)):
        cur = [0] * (n + 1)
        ai = a[i]
        for j in range(n):
            cur[j + 1] = prev[j] + 1 if ai == b[j] else max(cur[j], prev[j + 1])
        prev = cur
    return prev[n]


def rouge_l(pred, gold):
    a, b = tokens(pred), tokens(gold)
    if not a or not b:
        return 0.0
    length = lcs_len(a, b)
    return 2 * length / (len(a) + len(b)) if length else 0.0


def hallu(pred, context, gold):
    evidence = set(VER.findall(context or "")) | set(VER.findall(gold or ""))
    return 1 if any(v not in evidence for v in VER.findall(pred or "")) else 0


def refused(pred):
    return 1 if "not covered" in str(pred or "").lower() else 0


def three_sections(pred):
    low = str(pred or "").lower()
    return 1 if ("root cause" in low and "version" in low and "repair" in low) else 0


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/case_comparison_100.csv")
    ap.add_argument("--bare", default="results/bare_preds.csv")
    ap.add_argument("--data", default="data/bigdata_error_cases_500_en.csv")
    ap.add_argument("--out", default="results/metrics_recomputed.csv")
    args = ap.parse_args()

    ev = {str(r["id"]).strip(): r
          for r in csv.DictReader(open(args.data, encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(args.pred, encoding="utf-8-sig")))
    bare = {str(r["case_id"]).strip(): r
            for r in csv.DictReader(open(args.bare, encoding="utf-8-sig"))} \
        if os.path.exists(args.bare) else {}

    table, per_case = [], {}
    n = len(rows)
    for cfg in CONFIGS:
        col = f"{cfg}_pred"
        m = {"config": LABEL[cfg], "n": n}
        m["EM"] = sum(strict_em(r[col], r["gold"]) for r in rows)
        m["wordF1"] = sum(word_f1(r[col], r["gold"]) for r in rows) / n
        m["rougeL"] = sum(rouge_l(r[col], r["gold"]) for r in rows) / n
        hr = [hallu(r[col], ev[str(r["case_id"])]["context"], r["gold"]) for r in rows]
        m["version_halluc_rate"] = sum(hr) / n
        m["refuse_rate"] = sum(refused(r[col]) for r in rows) / n
        m["three_sections"] = sum(three_sections(r[col]) for r in rows) / n
        m["mean_len_words"] = sum(len(tokens(r[col])) for r in rows) / n
        lo, hi = wilson(m["EM"], n)
        m["EM_CI"] = f"[{lo:.3f},{hi:.3f}]"
        per_case[cfg] = {"f1": [word_f1(r[col], r["gold"]) for r in rows], "hallu": hr}
        table.append(m)

    if bare:
        m = {"config": "Bare LLM (zero-shot)", "n": len(bare)}
        ids = [str(r["case_id"]).strip() for r in rows]
        m["EM"] = sum(strict_em(bare[i]["pred"], bare[i]["gold"]) for i in ids)
        m["wordF1"] = sum(word_f1(bare[i]["pred"], bare[i]["gold"]) for i in ids) / len(ids)
        m["rougeL"] = sum(rouge_l(bare[i]["pred"], bare[i]["gold"]) for i in ids) / len(ids)
        m["version_halluc_rate"] = sum(
            hallu(bare[i]["pred"], "", bare[i]["gold"]) for i in ids) / len(ids)
        m["refuse_rate"] = sum(refused(bare[i]["pred"]) for i in ids) / len(ids)
        m["three_sections"] = sum(three_sections(bare[i]["pred"]) for i in ids) / len(ids)
        m["mean_len_words"] = sum(len(tokens(bare[i]["pred"])) for i in ids) / len(ids)
        lo, hi = wilson(m["EM"], len(ids))
        m["EM_CI"] = f"[{lo:.3f},{hi:.3f}]"
        per_case["bare"] = {"f1": [word_f1(bare[i]["pred"], bare[i]["gold"]) for i in ids],
                            "hallu": [hallu(bare[i]["pred"], "", bare[i]["gold"]) for i in ids]}
        table.insert(0, m)
        ids_bare = ids

    cols = ["config", "n", "EM", "EM_CI", "wordF1", "rougeL",
            "version_halluc_rate", "refuse_rate", "three_sections", "mean_len_words"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for m in table:
            row = dict(m)
            for k in ("wordF1", "rougeL", "version_halluc_rate", "refuse_rate",
                      "three_sections", "mean_len_words"):
                row[k] = f"{row[k]:.4f}"
            w.writerow(row)

    print(f"{'configuration':<26}{'EM':>7}{'F1':>9}{'ROUGE-L':>9}{'halluc':>9}{'refuse':>8}{'len':>8}")
    for m in table:
        print(f"{m['config']:<26}{m['EM']:>3}/{m['n']:<3}{m['wordF1']:>9.4f}"
              f"{m['rougeL']:>9.4f}{m['version_halluc_rate']:>9.4f}"
              f"{m['refuse_rate']:>8.2f}{m['mean_len_words']:>8.1f}")

    # Paired tests -> results/table4_paired.csv (backs Table 4 of the paper)
    paired = []
    print("\nMcNemar on version hallucination (exact binomial)")
    for a, b in (("bare", "baseline"), ("baseline", "pipeline")):
        if a not in per_case or b not in per_case:
            continue
        ha, hb = per_case[a]["hallu"], per_case[b]["hallu"]
        bo = sum(1 for x, y in zip(ha, hb) if x and not y)
        co = sum(1 for x, y in zip(ha, hb) if y and not x)
        p = mcnemar(bo, co)
        print(f"  {a:>8} -> {b:<8} b={bo:<3} c={co:<3} p={p:.3g}")
        paired.append({"comparison": f"{a} -> {b}", "metric": "version_hallucination",
                       "test": "McNemar exact binomial", "n_discordant_b": bo,
                       "n_discordant_c": co, "statistic": "", "p_value": f"{p:.3g}"})

    try:
        from scipy.stats import wilcoxon
        print("\nWilcoxon signed-rank on per-case word F1")
        for a, b in (("bare", "baseline"), ("baseline", "pipeline")):
            if a in per_case and b in per_case:
                stat, p = wilcoxon(per_case[a]["f1"], per_case[b]["f1"])
                print(f"  {a:>8} -> {b:<8} p={p:.3g}")
                paired.append({"comparison": f"{a} -> {b}", "metric": "word_f1",
                               "test": "Wilcoxon signed-rank", "n_discordant_b": "",
                               "n_discordant_c": "", "statistic": f"{stat:.3f}",
                               "p_value": f"{p:.3g}"})
    except Exception as exc:  # scipy missing or all-zero differences
        print(f"\n[skip] Wilcoxon unavailable: {exc}")

    if paired:
        out4 = os.path.join(os.path.dirname(os.path.abspath(args.out)), "table4_paired.csv")
        with open(out4, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["comparison", "metric", "test",
                                               "n_discordant_b", "n_discordant_c",
                                               "statistic", "p_value"])
            w.writeheader()
            w.writerows(paired)
        print(f"wrote {out4}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
