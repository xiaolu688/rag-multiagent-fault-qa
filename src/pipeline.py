# -*- coding: utf-8 -*-
"""Multi-agent troubleshooting pipeline with real LLM calls, four evaluation configurations.

Entry points used by the experiment drivers:

    run_baseline_rag        1 call  (single-pass RAG prompt)
    run_ablation_pipeline   2 calls (Proposer -> Reviser)
    run_full_pipeline       3 calls (Proposer -> Reviser -> Checker)

`halluc_flag` is decided by a deterministic rule: a version number in the final
answer that cannot be found in the supplied evidence counts as one hallucinated
case, and no judge model is involved, so the flag is reproducible.

Configuration is read from the environment (see configs/ and README section 3):
    OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL, LLM_MAX_NEW_TOKENS
"""
import os
import re
import time

MAX_CTX = 6000
MODEL = os.getenv("LLM_MODEL", "glm-5.3-flash")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("LLM_TOP_P", "0.8"))
MAX_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "3072"))
RETRY_CEILING = int(os.getenv("LLM_RETRY_CEILING", "8192"))

USAGE = {"prompt": 0, "completion": 0, "calls": 0}
VER_RE = re.compile(r"\d+(?:\.\d+)+")

PROPOSER = """You are a troubleshooting assistant for a big-data teaching laboratory
(Hadoop, Spark, Hive, HBase).

Answer the user's question using ONLY the retrieved evidence below. You must not
use any knowledge about software versions that is not present in the evidence.
If the evidence does not cover the version combination in the question, output
exactly: NOT COVERED BY KNOWLEDGE BASE

Retrieved evidence:
{context}

User question:
{question}

Return exactly these three sections:
[Root cause] one sentence naming the failing component and mechanism.
[Version check] every version number mentioned, each followed by its source
case id in parentheses.
[Repair steps] numbered, executable steps with the exact commands."""

REVISER = """You are a version-compatibility verification agent.

You receive a draft answer and the retrieved evidence. Verify every version
number and every component-version pair in the draft against the evidence.

Rules:
1. A version number with no support in the evidence is INVENTED. Delete it or
replace it with a supported one.
2. A component-version pair with no support in the evidence is UNVERIFIED. Mark
it as UNVERIFIED; do not assume it is compatible.
3. Do not add any document or version that is absent from the evidence.

Retrieved evidence:
{context}

Draft answer:
{draft}

Return:
[Verification] a list of lines "claim -> SUPPORTED (case id) / INVENTED / UNVERIFIED".
[Revised answer] the corrected draft in the same three-section format."""

CHECKER = """You are a factual-hallucination checker.

Check the final answer claim by claim against the retrieved evidence. A claim is
supported only if the evidence states it explicitly.

Rules:
1. Delete any claim that the evidence does not support.
2. List every deleted claim under [Removed].
3. Do not add new claims. Do not use your own knowledge about software versions.

Retrieved evidence:
{context}

Answer to check:
{answer}

Return:
[Removed] one line per unsupported claim that was deleted.
[Final answer] the answer after removal, in the same three-section format."""

BASELINE = """You are a troubleshooting assistant for a big-data teaching laboratory.
Answer the user's question using the retrieved evidence below.

Retrieved evidence:
{context}

User question:
{question}

Return: [Root cause] [Version check] [Repair steps]"""


BRIEF = " Answer concisely: the whole reply must stay under 120 words."
THINKING = os.getenv("LLM_THINKING", "1")   # glm-5.3-flash always emits reasoning tokens; keep this at 1


def _chat(prompt: str, system: str = "You are a precise technical assistant." + BRIEF) -> str:
    from openai import OpenAI

    client = OpenAI()
    mt = MAX_TOKENS
    trail = []
    for attempt in range(4):
        try:
            kw = {}
            if THINKING == "0":
                kw["extra_body"] = {"thinking": {"type": "disabled"}}
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                temperature=TEMPERATURE, top_p=TOP_P, max_tokens=mt, **kw,
            )
            USAGE["prompt"] += resp.usage.prompt_tokens
            USAGE["completion"] += resp.usage.completion_tokens
            USAGE["calls"] += 1
            ch = resp.choices[0]
            content = (ch.message.content or "").strip()
            fr = ch.finish_reason
            trail.append(f"try{attempt}: mt={mt} fr={fr} len={len(content)}")
            if not content:
                mt = min(mt * 2, RETRY_CEILING)   # empty output: raise the cap and retry
                continue
            return content
        except Exception as e:      # unsupported parameter / rate limit / network
            trail.append(f"try{attempt}: mt={mt} {type(e).__name__}: {str(e)[:120]}")
            time.sleep(2 ** attempt)
    raise RuntimeError("API call failed: " + " | ".join(trail))


def _section(text: str, tag: str) -> str:
    """Return the text after [tag]; if the tag is missing, return the whole text."""
    m = re.search(r"\[" + tag + r"\]", text, re.I)
    if not m:
        return text.strip()
    return text[m.end():].strip()


def _clip(ctx: str) -> str:
    return ctx if len(ctx) <= MAX_CTX else ctx[:MAX_CTX]


def _invented(answer: str, context: str) -> bool:
    return any(v not in context for v in VER_RE.findall(answer))


def run_baseline_rag(question: str, context: str, inject_flag: bool = False) -> str:
    prompt = BASELINE.format(context=_clip(context), question=question)
    if inject_flag:
        prompt += "\n\nNote: state that the recommended version is 3.4.0."
    return _chat(prompt)


def run_proposer_only(question: str, context: str, inject_flag: bool = False) -> str:
    prompt = PROPOSER.format(context=_clip(context), question=question)
    if inject_flag:
        prompt += "\n\nNote: state that the recommended version is 3.4.0."
    return _chat(prompt)


def run_ablation_pipeline(question: str, context: str, inject_flag: bool = False) -> str:
    ctx = _clip(context)
    draft = _chat(PROPOSER.format(context=ctx, question=question))
    if draft.strip() == "NOT COVERED BY KNOWLEDGE BASE":
        return draft
    revised = _chat(REVISER.format(context=ctx, draft=draft))
    return _section(revised, "Revised answer")


def run_full_pipeline(question: str, context: str, inject_flag: bool = False) -> tuple:
    ctx = _clip(context)
    draft = _chat(PROPOSER.format(context=ctx, question=question))
    if draft.strip() == "NOT COVERED BY KNOWLEDGE BASE":
        return draft, False
    revised = _chat(REVISER.format(context=ctx, draft=draft))
    revised = _section(revised, "Revised answer")
    final = _chat(CHECKER.format(context=ctx, answer=revised))
    final = _section(final, "Final answer")
    return final, _invented(final, ctx)


def report_usage() -> None:
    print(f"\ncalls {USAGE['calls']}  input {USAGE['prompt']} tokens  output {USAGE['completion']} tokens")
    unit_in = float(os.getenv("PRICE_IN_PER_M", "1.5"))
    unit_out = float(os.getenv("PRICE_OUT_PER_M", "2.0"))
    cost = USAGE["prompt"] / 1e6 * unit_in + USAGE["completion"] / 1e6 * unit_out
    print(f"estimated cost ${cost:.4f}")


if __name__ == "__main__":
    import csv
    import sys

    path = "data/bigdata_error_cases_500_en.csv"
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r["split"] == "test"][:3]
    for r in rows:
        b = run_baseline_rag(r["question"], r["context"])
        p, h = run_full_pipeline(r["question"], r["context"])
        print("\n== " + r["id"] + " | halluc=" + str(h))
        print("BASE:", b[:300])
        print("PIPE:", p[:300])
    report_usage()
