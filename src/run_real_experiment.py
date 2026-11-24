# -*- coding: utf-8 -*-
"""Experiment driver: 100 test cases x 4 configurations, with per-case predictions.

Run from the repository root:
    python src/run_real_experiment.py --limit 5     # smoke test on 5 cases
    python src/run_real_experiment.py               # full run (~700 model calls)

Outputs
    results/real_run_progress.jsonl  one line per finished case, used to resume
    results/case_comparison_100.csv  per-case predictions of the 4 configurations
    results/latency_cost.csv         latency, tokens and cost per configuration
    results/token_ledger.csv         per-case token accounting
    results/table3_real.csv          legacy strict-EM / Wilson-CI table
"""
import argparse
import csv
import math
import os
import json
import re
import time

import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config as _cfg
_cfg.apply()
import pipeline as mp

DATA = "data/bigdata_error_cases_500_en.csv"
PROG = "results/real_run_progress.jsonl"
CONFIGS = ["baseline", "proposer", "reviser", "pipeline"]
CFG_NAMES = {"baseline": "Baseline RAG", "proposer": "Proposer only",
             "reviser": "Proposer + Reviser", "pipeline": "Full pipeline"}
CALLS = {"baseline": 1, "proposer": 1, "reviser": 2, "pipeline": 3}
FIELDS = (["case_id", "component", "question", "gold"]
          + [f"{c}_pred" for c in CONFIGS]
          + [f"{c}_ms" for c in CONFIGS]
          + [f"{c}_in" for c in CONFIGS] + [f"{c}_out" for c in CONFIGS]
          + ["cost_usd"])


def norm(x) -> str:
    return re.sub(r"\s+", " ", str(x).lower()).strip()


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


def cost(pin, pout):
    unit_in = float(os.getenv("PRICE_IN_PER_M", "1.5"))
    unit_out = float(os.getenv("PRICE_OUT_PER_M", "2.0"))
    return pin / 1e6 * unit_in + pout / 1e6 * unit_out


def load_test(limit):
    rows = [{k.strip(): (v or "").strip() for k, v in r.items()}
            for r in csv.DictReader(open(DATA, encoding="utf-8-sig"))]
    test = [r for r in rows if r.get("split") == "test"]
    if not test:
        raise SystemExit("no rows with split == test")
    print(f"dataset {len(rows)} rows, {len(test)} test cases")
    return test[:limit] if limit else test


def run_one(cfg, q, ctx):
    if cfg == "baseline":
        return mp.run_baseline_rag(q, ctx, False)
    if cfg == "proposer":
        return mp.run_proposer_only(q, ctx, False)
    if cfg == "reviser":
        return mp.run_ablation_pipeline(q, ctx, False)
    return mp.run_full_pipeline(q, ctx, False)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    cases = load_test(args.limit)
    done = {}
    if os.path.exists(PROG):
        for line in open(PROG, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            done[str(r["case_id"])] = r
        print(f"{len(done)} cases already done, skipping")

    fh = open(PROG, "a", encoding="utf-8")

    for i, row in enumerate(cases, 1):
        cid = row["id"]
        if cid in done:
            continue
        q, ctx, gold = row["question"], row["context"], row["ground_truth"]
        out = {"case_id": cid, "component": row.get("component", ""),
               "question": q, "gold": gold}

        for cfg in CONFIGS:
            p0, c0 = mp.USAGE["prompt"], mp.USAGE["completion"]
            t0 = time.time()
            out[f"{cfg}_pred"] = run_one(cfg, q, ctx)
            out[f"{cfg}_ms"] = round((time.time() - t0) * 1000, 1)
            out[f"{cfg}_in"] = mp.USAGE["prompt"] - p0
            out[f"{cfg}_out"] = mp.USAGE["completion"] - c0

        tin = sum(int(out[f"{c}_in"]) for c in CONFIGS)
        tout = sum(int(out[f"{c}_out"]) for c in CONFIGS)
        out["cost_usd"] = round(cost(tin, tout), 6)

        fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        fh.flush()
        print(f"[{i}/{len(cases)}] id={cid} pipeline {out['pipeline_ms']:.0f}ms tokens={tin}/{tout}")

        if args.sleep:
            time.sleep(args.sleep)
    fh.close()

    rows = [json.loads(l) for l in open(PROG, encoding="utf-8") if l.strip()]
    raw = len(rows)
    rows = [r for r in rows if r.get("baseline_pred") and r.get("pipeline_pred")]
    print(f"\n{len(rows)} cases summarised ({raw} raw, {raw - len(rows)} empty predictions)")
    if not rows:
        print("no usable cases: empty predictions usually mean the output was cut off by max_tokens; raise LLM_MAX_NEW_TOKENS and rerun.")
        return

    with open("results/case_comparison_100.csv", "w", encoding="utf-8", newline="") as f:
        cols = ["case_id", "component", "question", "gold"] + [f"{c}_pred" for c in CONFIGS]
        fw = csv.DictWriter(f, fieldnames=cols)
        fw.writeheader()
        for r in rows:
            fw.writerow({k: r[k] for k in cols})

    with open("results/token_ledger.csv", "w", encoding="utf-8-sig", newline="") as f:
        fw = csv.writer(f)
        fw.writerow(["config", "case_id", "prompt_tokens", "completion_tokens", "cost_usd"])
        for r in rows:
            tin = sum(int(r[f"{c}_in"]) for c in CONFIGS)
            tout = sum(int(r[f"{c}_out"]) for c in CONFIGS)
            fw.writerow(["sum_of_4_configs", r["case_id"], tin, tout, r["cost_usd"]])

    em, lat_rows, table = {}, [], []
    for cfg in CONFIGS:
        v = [1 if norm(r[f"{cfg}_pred"]) == norm(r["gold"]) else 0 for r in rows]
        em[cfg] = v
        k, n = sum(v), len(v)
        lo, hi = wilson(k, n)
        ms = [float(r[f"{cfg}_ms"]) for r in rows]
        tin = sum(int(r[f"{cfg}_in"]) for r in rows)
        tout = sum(int(r[f"{cfg}_out"]) for r in rows)
        lat_rows.append({"config": CFG_NAMES[cfg], "n": n,
                         "mean_latency_ms": round(float(np.mean(ms)), 1),
                         "p95_latency_ms": round(float(np.percentile(ms, 95)), 1),
                         "llm_calls_per_case": CALLS[cfg], "total_llm_calls": CALLS[cfg] * n,
                         "prompt_tokens": tin, "completion_tokens": tout,
                         "cost_usd": round(cost(tin, tout), 6)})
        table.append({"config": CFG_NAMES[cfg], "EM": f"{k}/{n}", "rate": f"{k/n:.3f}",
                      "CI": f"[{lo:.3f}, {hi:.3f}]"})
        print(f"{CFG_NAMES[cfg]:22s} EM={k}/{n}={k/n:.3f} CI[{lo:.3f},{hi:.3f}] latency {np.mean(ms):.0f}ms tokens {tin}/{tout}")

    b, p = em["baseline"], em["pipeline"]
    both = sum(1 for x, y in zip(b, p) if x and y)
    imp = sum(1 for x, y in zip(b, p) if not x and y)
    deg = sum(1 for x, y in zip(b, p) if x and not y)
    wrong = sum(1 for x, y in zip(b, p) if not x and not y)
    print(f"\npaired: both correct {both} / improved {imp} / degraded {deg} / both wrong {wrong}  McNemar p={mcnemar(imp, deg):.4f}")

    tot_in = sum(sum(int(r[f"{c}_in"]) for c in CONFIGS) for r in rows)
    tot_out = sum(sum(int(r[f"{c}_out"]) for c in CONFIGS) for r in rows)
    tot_cost = sum(float(r["cost_usd"]) for r in rows)
    print(f"tokens: input {tot_in} / output {tot_out}  cost ${tot_cost:.4f}  per case ${tot_cost/len(rows):.5f}")

    with open("results/latency_cost.csv", "w", encoding="utf-8", newline="") as f:
        fw = csv.DictWriter(f, fieldnames=list(lat_rows[0].keys()))
        fw.writeheader()
        fw.writerows(lat_rows)

    with open("results/table3_real.csv", "w", encoding="utf-8", newline="") as f:
        fw = csv.DictWriter(f, fieldnames=["config", "EM", "rate", "CI"])
        fw.writeheader()
        fw.writerows(table)
    print("\nwrote case_comparison_100.csv, latency_cost.csv, token_ledger.csv, table3_real.csv")


if __name__ == "__main__":
    main()
