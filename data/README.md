# Dataset — `bigdata_error_cases_500_en`

## What it is

500 English troubleshooting cases for a big-data teaching laboratory, covering four
components with 125 cases each: **Hadoop, Spark, Hive, HBase**. Each case is a question
asked by a student, the documentation evidence that governs the answer, and the verified
repair procedure.

The dataset is built around **version compatibility**, because that is where a
retrieval-augmented system fails most silently: the same error string has a different root
cause under different component versions, and a model that "knows" a plausible version
number will answer fluently and wrongly.

## Schema

| column | meaning |
|---|---|
| `id` | integer id, 1–500 |
| `component` | `hadoop` / `spark` / `hive` / `hbase` |
| `question` | the student's question, as asked |
| `context` | evidence span (≈190 characters) from the documentation that governs the case |
| `ground_truth` | verified repair procedure, 3–5 numbered steps |
| `source` | URL of the governing document (Apache docs, Apache JIRA, Stack Overflow) or the course laboratory fault log |
| `split` | `kb` (retrieval knowledge base, 400) or `test` (held-out evaluation set, 100) |
| `status` | `verified` for all rows |

## Split

* 400 `kb` / 100 `test`, stratified 8:2 per component (100/25 each), frozen seed 42.
* No question and no evidence span appears on both sides:
  `python src/preprocessing.py` prints `test_question_leak: 0`, `test_evidence_leak: 0`
  and writes `data/split_report.json`.

## How to obtain / regenerate

The CSV is committed, so nothing needs downloading. To regenerate the split:

```bash
python src/preprocessing.py --rebuild     # re-derives the split in memory, writes data/split_report.json only
```

The shipped split is frozen in the CSV: `--rebuild` re-runs the same stratification procedure
as a check and never overwrites the data file.

## How it is used in the experiments

The reported experiments feed the case's own `context` to the agents
(`retrieval.mode: oracle_evidence`). This keeps the comparison between the three agents free
of retrieval noise and keeps the split leak-free. The BM25 + Chroma retriever is a design
component and is not queried at evaluation time.

## Licensing of the source material

`question` and `ground_truth` were written by the author; `context` paraphrases the cited
documentation and each row links to its `source` URL. No copyrighted text is redistributed
beyond short evidence spans with attribution.
