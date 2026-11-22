# -*- coding: utf-8 -*-
"""Recompute the version-hallucination rate and the McNemar test (Appendix C.5 definition).

A version number in the answer that is absent from the version set of
"evidence passage u gold answer" counts as one hallucinated case; an answer that
names no version is not counted.

Usage (from the repository root):
    python src/evaluate_hallucination.py
"""
import re, csv, sys, os
from math import comb

VER = re.compile(r"\b\d+\.\d+(?:\.\d+){0,2}\b")
DATASET = "bigdata_error_cases_500_en.csv"

def find(name, default_dir):
    for base in (os.getcwd(), "..", default_dir):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    sys.exit(f"cannot find {name}")

def is_halluc(pred, ctx, gold):
    evidence = set(VER.findall(ctx)) | set(VER.findall(gold))
    return any(v not in evidence for v in VER.findall(pred or ""))

ds = {str(r["id"]).strip(): r
      for r in csv.DictReader(open(find(DATASET, "data"), encoding="utf-8-sig"))}
rows = list(csv.DictReader(open(find("case_comparison_100.csv", "results"), encoding="utf-8-sig")))
bare = {str(r["case_id"]).strip(): r
        for r in csv.DictReader(open(find("bare_preds.csv", "results"), encoding="utf-8-sig"))}

H = {}
for r in rows:
    cid = str(r["case_id"]).strip(); d = ds[cid]
    H[cid] = {
        "bare":     is_halluc(bare[cid]["pred"], "", d["ground_truth"]),
        "rag":      is_halluc(r["baseline_pred"], d["context"], d["ground_truth"]),
        "proposer": is_halluc(r["proposer_pred"], d["context"], d["ground_truth"]),
        "reviser":  is_halluc(r["reviser_pred"],  d["context"], d["ground_truth"]),
        "full":     is_halluc(r["pipeline_pred"], d["context"], d["ground_truth"]),
    }

print(f"{'config':<12}{'hallucinated':>13}{'rate':>9}")
rates = {}
for k in ("bare", "rag", "proposer", "reviser", "full"):
    h = sum(H[c][k] for c in H); rates[k] = h / len(H)
    print(f"{k:<12}{h:>13}{rates[k]:>9.3f}")

def mcnemar(a, b):
    bo = sum(1 for c in H if H[c][a] and not H[c][b])
    co = sum(1 for c in H if H[c][b] and not H[c][a])
    n = bo + co
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(bo, co) + 1)) / 2 ** n) if n else 1.0
    return bo, co, p

print("\nMcNemar (exact binomial)")
for a, b in (("bare", "rag"), ("rag", "full"), ("bare", "full")):
    bo, co, p = mcnemar(a, b)
    print(f"{a:>6} → {b:<6} b={bo:<3} c={co:<3} p={p:.3g}")

with open("results/metrics_halluc.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["config", "n", "halluc_count", "version_halluc_rate"])
    for k in ("bare", "rag", "proposer", "reviser", "full"):
        w.writerow([k, len(H), sum(H[c][k] for c in H), f"{rates[k]:.4f}"])
print("\nwrote results/metrics_halluc.csv")
