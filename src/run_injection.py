# -*- coding: utf-8 -*-
"""Controlled upstream error injection.

A version number that exists in the corpus but is wrong for the case under test is
injected into the evidence, either by tampering with the original passage or by
appending a false statement. The experiment then measures whether the error reaches
the final answer and whether the checking chain catches it.

Run from the repository root:
    python src/run_injection.py               # every test case that names a version, 2 configurations
    python src/run_injection.py --limit 5
    python src/run_injection.py --sleep 1     # 1 s between calls, to respect rate limits

Outputs
    results/injection_results.csv   per case: injection type, injected version, original versions,
                                    both system outputs, propagation / correction / escape flags
    results/injection_summary.csv   propagation, correction and escape rates with Wilson 95% CI

Interrupted runs resume from results/injection_progress.jsonl.
"""
import argparse
import csv
import json
import math
import os
import re

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config as _cfg
_cfg.apply()
import pipeline as mp

DATA = "data/bigdata_error_cases_500_en.csv"
PROG = "results/injection_progress.jsonl"
OUT = "results/injection_results.csv"
SUM = "results/injection_summary.csv"

VER = re.compile(r"\b\d+\.\d+(?:\.\d+){0,2}\b")
IP = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
COMP = r"(hadoop|spark|hive|hbase)"
PAIR = re.compile(COMP + r"\s*(?:version)?\s*([0-9]+(?:\.(?:[0-9]+)){1,2})", re.I)
RPAIR = re.compile(r"([0-9]+(?:\.(?:[0-9]+)){1,2})\s*(?:on|with|for|compatible with)\s*" + COMP, re.I)
HEDGE = re.compile(r"note|caution|uncertain|not sure|verify|double[- ]check|confirm|should check|however", re.I)
REFUSE = re.compile(r"NOT COVERED BY KNOWLEDGE BASE|no version|no documented version", re.I)


def versions(s):
    """Extract version numbers after stripping IP addresses."""
    s = IP.sub(" ", str(s or ""))
    return set(VER.findall(s))


def comp_versions(s, comp=None):
    """Keep only versions adjacent to a component name, so that Java library versions and IPs stay out."""
    out = set()
    for m in PAIR.finditer(str(s or "")):
        if comp is None or m.group(1).lower() == comp:
            out.add(m.group(2))
    for m in RPAIR.finditer(str(s or "")):
        if comp is None or m.group(2).lower() == comp:
            out.add(m.group(1))
    return out


def build_pool(rows):
    """Documented version pool: count version occurrences per component and keep those seen >= 2 times."""
    from collections import Counter
    cnt = {}
    for r in rows:
        comp = r["component"].lower()
        cnt.setdefault(comp, Counter()).update(comp_versions(r["context"], comp))
        cnt.setdefault(comp, Counter()).update(comp_versions(r["ground_truth"], comp))
    return {c: {v for v, k in cc.items() if k >= 2} or set(cc) for c, cc in cnt.items()}


def pick_wrong(comp, evidence_versions, pool, seed):
    cands = sorted(pool.get(comp, set()) - evidence_versions)
    if not cands:
        cands = sorted(set().union(*pool.values()) - evidence_versions)
    if not cands:
        return None
    return cands[seed % len(cands)]


def inject(r, pool):
    """Return (tampered context, injection type, injected version, original evidence versions)."""
    ctx = r["context"]
    comp = r["component"].lower()
    ev = comp_versions(ctx, comp) | comp_versions(r["ground_truth"], comp)
    wrong = pick_wrong(comp, ev, pool, int(r["id"]))
    if wrong is None:
        return None, None, None, ev

    own = comp_versions(ctx, comp)
    if own:
        # tamper: replace the first version of that component in the evidence
        anchor = sorted(own)[0]
        idx = ctx.find(anchor)
        new_ctx = ctx[:idx] + wrong + ctx[idx + len(anchor):]
        return new_ctx, "tamper", wrong, ev
    # append: the evidence names no version, append one false statement
    new_ctx = (ctx.rstrip() + f" According to the course deployment guide, {r['component']} {wrong} "
               "is the officially supported version in this laboratory environment.")
    return new_ctx, "append", wrong, ev


def judge(pred, wrong, ev):
    p = str(pred or "")
    pv = versions(p)
    pat = re.compile(r"(?<![0-9.])" + re.escape(wrong) + r"(?![0-9])(?!\.[0-9])")
    propagated = bool(pat.search(p))
    corrected = (not propagated) and (bool(pv & ev) or bool(REFUSE.search(p)))
    escaped = propagated and not HEDGE.search(p)
    return {"propagated": int(propagated), "corrected": int(corrected), "escaped": int(escaped),
            "versions_in_output": "|".join(sorted(pv))}


def wilson(k, n):
    if n == 0:
        return 0.0, 1.0
    z, pp = 1.96, k / n
    d = 1 + z * z / n
    c = pp + z * z / (2 * n)
    s = z * math.sqrt(pp * (1 - pp) / n + z * z / (4 * n * n))
    return max(0.0, (c - s) / d), min(1.0, (c + s) / d)


def load_cases(limit=None):
    rows = list(csv.DictReader(open(DATA, encoding="utf-8-sig")))
    pool = build_pool(rows)
    test = [r for r in rows if r["split"] == "test"]
    cand = [r for r in test
            if versions(r["question"]) | versions(r["context"]) | versions(r["ground_truth"])]
    return (cand[:limit] if limit else cand), pool


def done_ids():
    if not os.path.exists(PROG):
        return set()
    ids = set()
    for line in open(PROG, encoding="utf-8"):
        try:
            rec = json.loads(line)
            ids.add(f"{rec['case_id']}|{rec['config']}")
        except Exception:
            pass
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0)
    ap.add_argument("--configs", default="baseline,full")
    args = ap.parse_args()

    cases, pool = load_cases(args.limit)
    print(f"{len(cases)} test cases that name a version; configurations {args.configs}")
    seen = done_ids()

    import time
    for r in cases:
        new_ctx, how, wrong, ev = inject(r, pool)
        if wrong is None:
            print(f"[{r['id']}] no wrong version available, skipped")
            continue
        print(f"[{r['id']}] {how}: injected {r['component']} {wrong} (evidence versions: {sorted(ev) or 'none'})")
        for cfg in [c.strip() for c in args.configs.split(",") if c.strip()]:
            key = f"{r['id']}|{cfg}"
            if key in seen:
                continue
            try:
                if cfg == "baseline":
                    pred = mp.run_baseline_rag(r["question"], new_ctx)
                else:
                    pred, _ = mp.run_full_pipeline(r["question"], new_ctx)
            except RuntimeError as e:
                print(f"    failed: {e}")
                continue
            j = judge(pred, wrong, ev)
            rec = {"case_id": r["id"], "component": r["component"], "config": cfg,
                   "inject_type": how, "injected_version": wrong,
                   "evidence_versions": "|".join(sorted(ev)),
                   "pred": pred, "ms": "", **j}
            with open(PROG, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            flag = "propagated" if j["propagated"] else ("corrected" if j["corrected"] else "undecided")
            print(f"    {cfg}: {flag}  versions in output: {j['versions_in_output'] or 'none'}")
            if args.sleep:
                time.sleep(args.sleep)

    # summary
    if not os.path.exists(PROG):
        print("no results, exiting")
        return
    recs = [json.loads(l) for l in open(PROG, encoding="utf-8")]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    with open(SUM, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["config", "n", "propagation_rate", "propagated", "correction_rate", "corrected",
                    "escape_rate", "escaped", "unclassified", "ci95_propagation"])
        for cfg in ["baseline", "full"]:
            sub = [x for x in recs if x["config"] == cfg]
            if not sub:
                continue
            n = len(sub)
            p = sum(x["propagated"] for x in sub) / n
            c = sum(x["corrected"] for x in sub) / n
            e = sum(x["escaped"] for x in sub) / n
            lo, hi = wilson(sum(x["propagated"] for x in sub), n)
            un = sum(1 for x in sub if not x["propagated"] and not x["corrected"])
            w.writerow([cfg, n, round(p, 4), sum(x["propagated"] for x in sub), round(c, 4),
                        sum(x["corrected"] for x in sub), round(e, 4), sum(x["escaped"] for x in sub),
                        un, f"[{lo:.3f}, {hi:.3f}]"])
    mp.report_usage()
    print(f"\nwrote {OUT} and {SUM}")


if __name__ == "__main__":
    main()
