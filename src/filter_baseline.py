# -*- coding: utf-8 -*-
"""Deterministic structured version filter (rule-based baseline, zero LLM calls).

The single-pass RAG answer is used as the candidate. Pure rules (regex plus a version
lookup) delete or replace version numbers that the evidence does not support. This
baseline exists to test whether multi-agent verification is needed on top of rules.

Judgement, identical to Table 2 of the paper:
  2 = at least one version is given and every version appears in the case evidence
      (context u gold)
  1 = no version is given (including an explicit disclaimer)
  0 = at least one version appears that the evidence does not contain

A separate "documented version" vocabulary is built from the context and gold of all
500 cases (including the official HBase compatibility matrix replicated in the corpus).
It splits unsupported versions into "exists in the corpus but not in this case's
evidence" and "fabricated outright".
"""
import csv, json, re, math
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as _cfg
_cfg.apply()
import pipeline as mp  # noqa: F401  (kept for parity; this baseline makes zero LLM calls)
from collections import Counter

DS = "data/bigdata_error_cases_500_en.csv"
RES = "results/"

VER = re.compile(r"\b\d+\.\d+(?:\.\d+){0,2}\b")
REFUSE = re.compile(r"NOT COVERED BY KNOWLEDGE BASE|not covered by knowledge|no version|no documented version|none\b", re.I)


def versions(s):
    return set(VER.findall(str(s or "")))


def load():
    rows = list(csv.DictReader(open(DS, encoding="utf-8-sig")))
    test = [r for r in rows if r["split"] == "test"]
    comp = {r["case_id"]: r for r in csv.DictReader(open(RES + "case_comparison_100.csv", encoding="utf-8-sig"))}
    bare = {r["case_id"]: r for r in csv.DictReader(open(RES + "bare_preds.csv", encoding="utf-8-sig"))}
    return rows, test, comp, bare


def documented_versions(all_rows):
    """Documented version vocabulary of the whole dataset (context + gold)."""
    v = set()
    for r in all_rows:
        v |= versions(r["context"]) | versions(r["ground_truth"])
    return v


def judge(pred, ev):
    """Return (vc label, versions given, unsupported versions)."""
    pv = versions(pred)
    if not pv:
        return 1, pv, set()
    bad = pv - ev
    return (0 if bad else 2), pv, bad


def filter_answer(pred, ev):
    """Rule filtering: replace an unsupported version with one the evidence carries, else drop it."""
    if not versions(pred):
        return pred, 0, 0
    out = pred
    replaced = removed = 0
    for v in sorted(versions(pred)):
        if v in ev:
            continue
        if ev:
            # take any version the evidence offers (no combinatorial reasoning: a known limit of rule filtering)
            out = out.replace(v, sorted(ev)[0])
            replaced += 1
        else:
            out = out.replace(v, "no documented version")
            removed += 1
    return out, replaced, removed


def wilson(k, n):
    if n == 0:
        return 0.0, 1.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0, (c - s) / d), min(1, (c + s) / d)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1, 2 * tail)


def main():
    all_rows, test, comp, bare = load()
    doc = documented_versions(all_rows)
    print("documented version vocabulary size:", len(doc))

    out = []
    for r in test:
        cid = r["id"]
        ev = versions(r["context"]) | versions(r["ground_truth"])
        rec = {"case_id": cid, "component": r["component"],
               "evidence_versions": "|".join(sorted(ev)),
               "question_versions": "|".join(sorted(versions(r["question"])))}
        cands = {
            "bare": bare[cid]["pred"],
            "baseline_rag": comp[cid]["baseline_pred"],
            "proposer": comp[cid]["proposer_pred"],
            "reviser": comp[cid]["reviser_pred"],
            "full_pipeline": comp[cid]["pipeline_pred"],
        }
        fp, rep, rem = filter_answer(cands["baseline_rag"], ev)
        cands["filter"] = fp
        rec["filter_replaced"], rec["filter_removed"] = rep, rem

        for name, pred in cands.items():
            vc, pv, bad = judge(pred, ev)
            rec[f"{name}_vc"] = vc
            rec[f"{name}_versions"] = "|".join(sorted(pv))
            rec[f"{name}_bad"] = "|".join(sorted(bad))
            # unsupported versions that are documented elsewhere in the corpus
            rec[f"{name}_bad_but_documented"] = "|".join(sorted(bad & doc))
        out.append(rec)

    with open(RES + "version_filter_eval.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    systems = ["bare", "baseline_rag", "proposer", "reviser", "full_pipeline", "filter"]
    summary = {}
    for s in systems:
        vcs = [x[f"{s}_vc"] for x in out]
        hall = sum(1 for v in vcs if v == 0)
        gave = sum(1 for v in vcs if v in (0, 2))
        cons = sum(1 for v in vcs if v == 2)
        refuse = sum(1 for v in vcs if v == 1)
        sub15 = [x for x in out if x["evidence_versions"]]
        sub30 = [x for x in out if x["evidence_versions"] or x["question_versions"]]
        rec15 = sum(1 for x in sub15 if x[f"{s}_vc"] == 2)
        rec30 = sum(1 for x in sub30 if x[f"{s}_vc"] == 2)
        lo, hi = wilson(hall, len(out))
        summary[s] = {"hallucination_rate": hall / len(out), "hall_cnt": hall,
                      "ci95": [round(lo, 4), round(hi, 4)],
                      "gave_version": gave, "consistent": cons, "no_version": refuse,
                      "recall_evidence15": f"{rec15}/{len(sub15)}", "recall_version30": f"{rec30}/{len(sub30)}"}

    # binary hallucination McNemar: full vs filter, baseline vs filter, bare vs filter
    def pairs(a, b):
        x = sum(1 for r0 in out if r0[f"{a}_vc"] == 0 and r0[f"{b}_vc"] != 0)
        y = sum(1 for r0 in out if r0[f"{a}_vc"] != 0 and r0[f"{b}_vc"] == 0)
        return x, y, mcnemar(x, y)

    summary["_mcnemar"] = {
        "baseline_vs_filter": pairs("baseline_rag", "filter"),
        "full_vs_filter": pairs("full_pipeline", "filter"),
        "bare_vs_baseline": pairs("bare", "baseline_rag"),
    }
    # share of unsupported versions that are documented elsewhere
    for s in ["bare", "baseline_rag", "full_pipeline"]:
        tot_bad = sum(len([v for v in (x[f"{s}_bad"] or "").split("|") if v]) for x in out)
        doc_bad = sum(len([v for v in (x[f"{s}_bad_but_documented"] or "").split("|") if v]) for x in out)
        summary[s]["bad_versions_total"] = tot_bad
        summary[s]["bad_versions_documented"] = doc_bad
        summary[s]["bad_versions_fabricated"] = tot_bad - doc_bad

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    json.dump(summary, open(RES + "version_filter_summary.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
