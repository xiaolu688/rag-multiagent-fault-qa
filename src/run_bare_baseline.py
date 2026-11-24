# -*- coding: utf-8 -*-
"""Bare-LLM baseline: zero-shot answer with no retrieved evidence.

Run from the repository root:
    python src/run_bare_baseline.py --limit 5
    python src/run_bare_baseline.py

Outputs
    results/bare_preds.csv          per-case predictions (case_id, question, gold, pred, ms, in, out)
    results/metrics_with_bare.csv   unified metric table (bare, single-pass RAG, and full pipeline)

Interrupted runs resume from results/bare_progress.jsonl.
"""
import argparse
import csv
import json
import math
import os
import re
import time

import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config as _cfg
_cfg.apply()
import pipeline as mp

DATA = "data/bigdata_error_cases_500_en.csv"
PROG = "results/bare_progress.jsonl"
OUT = "results/bare_preds.csv"
COMP = "results/case_comparison_100.csv"

SYSTEM = ("You are a helpful assistant for a big-data teaching laboratory. "
          "Answer the student's troubleshooting question directly from your own knowledge. "
          "Structure the answer as [Root cause], [Version check], [Repair steps].")

TOK = re.compile(r"[a-z0-9]+")
VER = re.compile(r"\b\d+\.\d+(?:\.\d+){0,2}\b")


def toks(s):
    return TOK.findall(str(s).lower())


def f1(gold, pred):
    G, P = set(toks(gold)), set(toks(pred))
    if not G or not P:
        return 0.0
    ov = len(G & P)
    if not ov:
        return 0.0
    pr, rc = ov / len(P), ov / len(G)
    return 2 * pr * rc / (pr + rc)


def lcs_f(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


def rouge_l(gold, pred):
    a, b = toks(gold), toks(pred)
    if not a or not b:
        return 0.0
    l = lcs_f(a, b)
    pr, rc = l / len(b), l / len(a)
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def wilson(k, n):
    if n == 0:
        return 0.0, 1.0
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - s) / d), min(1.0, (c + s) / d)


def load_cases(limit=None):
    rows = list(csv.DictReader(open(DATA, encoding="utf-8-sig")))
    cases = [r for r in rows if r.get("split") == "test"]
    return cases[:limit] if limit else cases


def run_one(question):
    """Three-step fallback: plain question -> forced short answer -> refusal allowed. Returns (answer, status)."""
    base = f"Question from a student: {question}"
    tries = [
        (base + "\n\nGive your best troubleshooting answer.", SYSTEM, "ok"),
        (base + "\n\nAnswer in at most three sentences, starting with [Root cause]. Do not deliberate.",
         SYSTEM, "brief"),
        (base + "\n\nIf the evidence is insufficient, reply exactly: NOT COVERED BY KNOWLEDGE BASE. "
                "Otherwise answer in at most five sentences.", SYSTEM, "fallback"),
    ]
    for prompt, system, status in tries:
        try:
            return mp._chat(prompt, system), status
        except RuntimeError:
            continue
    return "", "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0)
    args = ap.parse_args()

    cases = load_cases(args.limit)
    print(f"dataset {len(list(csv.DictReader(open(DATA, encoding='utf-8-sig'))))} rows, {len(cases)} test cases (bare model)")

    done = {}
    if os.path.exists(PROG):
        for line in open(PROG, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                done[str(r["case_id"])] = r
        print(f"{len(done)} cases already done, skipping")

    fh = open(PROG, "a", encoding="utf-8")
    for i, row in enumerate(cases, 1):
        cid = str(row["id"])
        if cid in done:
            continue
        p0, c0 = mp.USAGE["prompt"], mp.USAGE["completion"]
        t0 = time.time()
        pred, status = run_one(row["question"])
        out = {"case_id": cid, "status": status, "component": row.get("component", ""),
               "question": row["question"], "gold": row["ground_truth"],
               "pred": pred, "ms": round((time.time() - t0) * 1000, 1),
               "in": mp.USAGE["prompt"] - p0, "out": mp.USAGE["completion"] - c0}
        fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        fh.flush()
        print(f"[{i}/{len(cases)}] id={cid} {out['ms']:.0f}ms tokens={out['in']}/{out['out']}")
        if args.sleep:
            time.sleep(args.sleep)
    fh.close()

    rows = [json.loads(l) for l in open(PROG, encoding="utf-8") if l.strip()]
    raw = len(rows)
    failed = [r for r in rows if not r.get("pred")]
    rows = [r for r in rows if r.get("pred")]
    print(f"\n{len(rows)} cases summarised ({raw} raw, {len(failed)} failed)")
    if failed:
        print("failed case ids: ", ", ".join(str(r["case_id"]) for r in failed))

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "component", "question", "gold", "pred", "ms", "in", "out"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    # ---- metrics ----
    # Bare-LLM has no context, so its evidence set is the reference answer only.
    # The evidence-based configurations use context U gold, matching the paper's
    # Appendix C.5 definition; the question is not treated as evidence.
    ctx = {}
    for r in csv.DictReader(open(DATA, encoding="utf-8-sig")):
        ctx[str(r["id"])] = str(r["context"])

    def metrics(preds, golds, ids, ms=None, tin=None, tout=None, include_context=True):
        F = [f1(g, p) for g, p in zip(golds, preds)]
        R = [rouge_l(g, p) for g, p in zip(golds, preds)]
        hal = 0
        for p, k, g in zip(preds, ids, golds):
            ev = set(VER.findall(str(g)))
            if include_context:
                ev |= set(VER.findall(ctx.get(k, "")))
            if any(v not in ev for v in VER.findall(str(p))):
                hal += 1
        ref = sum(1 for p in preds if "NOT COVERED" in str(p).upper())
        sec = sum(1 for p in preds if all(s in str(p) for s in
                                          ["Root cause", "Version check", "Repair steps"]))
        k = sum(1 for g, p in zip(golds, preds) if str(p).strip() == str(g).strip())
        n = len(preds)
        lo, hi = wilson(k, n)
        return dict(n=n, EM=f"{k}/{n}", EM_rate=round(k / n, 4), EM_CI=f"[{lo:.3f},{hi:.3f}]",
                    wordF1=round(float(np.mean(F)), 4), rougeL=round(float(np.mean(R)), 4),
                    version_halluc_rate=round(hal / n, 4), refuse_rate=round(ref / n, 4),
                    three_sections=round(sec / n, 4),
                    mean_len_words=round(float(np.mean([len(toks(p)) for p in preds])), 1),
                    mean_latency_ms=(round(float(np.mean(ms)), 1) if ms else None),
                    prompt_tokens=(int(np.sum(tin)) if tin else None),
                    completion_tokens=(int(np.sum(tout)) if tout else None))

    table = [dict(config="Bare LLM (zero-shot)", **metrics(
        [r["pred"] for r in rows], [r["gold"] for r in rows],
        [r["case_id"] for r in rows], [float(r["ms"]) for r in rows],
        [int(r["in"]) for r in rows], [int(r["out"]) for r in rows], include_context=False))]

    if os.path.exists(COMP):
        d = list(csv.DictReader(open(COMP, encoding="utf-8")))
        lat = {}
        if os.path.exists("results/latency_cost.csv"):
            for x in csv.DictReader(open("results/latency_cost.csv", encoding="utf-8")):
                lat[x["config"]] = x
        for c, name, lat_name in [("baseline", "Single RAG (oracle ctx)", "Baseline RAG"),
                                  ("pipeline", "Full multi-agent", "Full pipeline")]:
            l = lat.get(lat_name, {})
            table.append(dict(config=name, **metrics(
                [r[f"{c}_pred"] for r in d], [r["gold"] for r in d], [r["case_id"] for r in d],
                ms=[float(l["mean_latency_ms"])] if l else None,
                tin=[int(l["prompt_tokens"])] if l else None,
                tout=[int(l["completion_tokens"])] if l else None)))

    cols = ["config", "n", "EM", "EM_rate", "EM_CI", "wordF1", "rougeL",
            "version_halluc_rate", "refuse_rate", "three_sections",
            "mean_len_words", "mean_latency_ms", "prompt_tokens", "completion_tokens"]
    with open("results/metrics_with_bare.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in table:
            w.writerow(t)

    print(f"\n{'config':26s} {'EM':>7s} {'wordF1':>7s} {'rL':>6s} {'halluc':>7s} {'refuse':>5s}")
    for t in table:
        print(f"{t['config']:26s} {t['EM']:>7s} {t['wordF1']:7.4f} {t['rougeL']:6.4f} "
              f"{t['version_halluc_rate']:7.2f} {t['refuse_rate']:5.2f}")

    print(f"\nwrote {OUT} and metrics_with_bare.csv")
    if mp.USAGE["calls"]:
        print(f"{mp.USAGE['calls']} calls, {mp.USAGE['prompt']} input / {mp.USAGE['completion']} output tokens")


if __name__ == "__main__":
    main()
