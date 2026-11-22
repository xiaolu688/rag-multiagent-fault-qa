# -*- coding: utf-8 -*-
"""Load a YAML experiment config into environment variables.

The pipeline reads its hyper-parameters from the environment (LLM_MODEL,
LLM_TEMPERATURE, ...), so a config file is applied once, before `pipeline`
is imported. Precedence: existing environment variables win over the YAML,
which lets you override a single value from the shell:

    RAG_CONFIG=configs/baseline.yaml python src/run_real_experiment.py --limit 5
    LLM_MAX_NEW_TOKENS=8192 python src/run_real_experiment.py
"""
import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "configs", "proposed_model.yaml")


def apply(name=None):
    path = name or os.environ.get("RAG_CONFIG")
    if not path:
        path = DEFAULT
    elif not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(ROOT, path)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    model = cfg.get("model", {}) or {}
    env = {
        "LLM_MODEL": model.get("name", "glm-5.3-flash"),
        "LLM_TEMPERATURE": model.get("temperature", 0.1),
        "LLM_TOP_P": model.get("top_p", 0.8),
        "LLM_MAX_NEW_TOKENS": model.get("max_new_tokens", 3072),
        "OPENAI_BASE_URL": model.get("base_url", "https://open.bigmodel.cn/api/paas/v4"),
        "LLM_RETRY_CEILING": model.get("empty_output_retry_tokens", 8192),
    }
    for key, val in env.items():
        os.environ.setdefault(key, str(val))
    for key, val in (cfg.get("pricing") or {}).items():
        os.environ.setdefault(key, str(val))
    cfg["_path"] = path
    return cfg
