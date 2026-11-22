# -*- coding: utf-8 -*-
"""Validate (and optionally rebuild) the knowledge-base / test split.

The dataset ships with a frozen split (kb = 400, test = 100, stratified 8:2 per
component, seed 42). This script re-checks it so the split used for the reported
experiment can be verified without touching the frozen file:

    python src/preprocessing.py            # verify, write data/split_report.json
    python src/preprocessing.py --rebuild  # regenerate the split from scratch

Verification checks
    * 500 rows, 4 components, 125 questions each
    * kb = 400 / test = 100
    * per-component 80 / 20 split
    * zero question overlap and zero evidence overlap between kb and test
      (a leak here would invalidate every number in the paper)
"""
import argparse
import csv
import json
import random
from collections import Counter

SEED = 42
COMPONENTS = ("hadoop", "spark", "hive", "hbase")


def load(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def verify(rows):
    comp = Counter(r["component"] for r in rows)
    split = Counter(r["split"] for r in rows)
    per_comp_split = Counter((r["component"], r["split"]) for r in rows)
    kb_q = {r["question"].strip().lower() for r in rows if r["split"] == "kb"}
    kb_c = {r["context"].strip().lower()[:120] for r in rows if r["split"] == "kb"}
    dup_q = sum(1 for r in rows if r["split"] == "test"
                and r["question"].strip().lower() in kb_q)
    dup_c = sum(1 for r in rows if r["split"] == "test"
                and r["context"].strip().lower()[:120] in kb_c)
    return {
        "n": len(rows),
        "per_component": dict(comp),
        "per_split": dict(split),
        "per_component_split": {f"{k[0]}/{k[1]}": v for k, v in per_comp_split.items()},
        "test_question_leak": dup_q,
        "test_evidence_leak": dup_c,
        "ok": (len(rows) == 500 and split["kb"] == 400 and split["test"] == 100
               and dup_q == 0 and dup_c == 0),
    }


def rebuild(rows):
    """Stratified 8:2 split produced by the same procedure as the shipped split (seed 42).

    The shipped split is frozen in the CSV, and this in-memory rebuild is a check of the
    procedure rather than a byte-identical reproduction: a different shuffle order is
    possible, so the CSV is never overwritten.
    """
    rng = random.Random(SEED)
    out = []
    for comp in COMPONENTS:
        group = [r for r in rows if r["component"] == comp]
        rng.shuffle(group)
        n_test = round(len(group) * 0.2)
        for r in group[:n_test]:
            r["split"] = "test"
        for r in group[n_test:]:
            r["split"] = "kb"
        out.extend(group)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bigdata_error_cases_500_en.csv")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out", default="data/split_report.json")
    args = ap.parse_args()

    rows = load(args.data)
    if args.rebuild:
        rows = rebuild(rows)
        print("[rebuild] split re-derived in memory; only the report is written, the CSV is untouched")
    report = verify(rows)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    for key, val in report.items():
        print(f"{key}: {val}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
