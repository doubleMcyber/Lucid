"""Export a *matched Python baseline* corpus from the same Loom records used for
Lucid, so H1–H4 compare languages, not task distributions.

Each record's Lucid program is transpiled to behaviorally-equivalent Python
(`loom.transpile_py`); the prompt (spec / IO + signature) is identical in intent
to the Lucid export, only the target language differs. The train/test split uses
the *same* `ast_hash` seeding and the *same* prompt-level leakage drop and
held-out-IO discipline as `loom.export`, so a program is on the same side in both
corpora and the comparison is fair and leakage-free.
"""

from __future__ import annotations

import json
import os

from loom.export import PROMPT_K, _fmt_io, _is_test, iter_records
from loom.transpile_py import entry_name, py_signature, to_python
from lucid.parser import parse

PY_CODE_TAG = "### Python:\n"


def build_prompt_py(task: str, rec: dict, py_sig: str) -> str:
    if task == "spec_to_code":
        spec = rec.get("spec_structured", "")
        return f"### Spec:\n{spec}\n### Signature:\n{py_sig}\n{PY_CODE_TAG}"
    if task == "io_to_code":
        io = _fmt_io(rec.get("io_examples", []))
        return f"### Examples (input -> output):\n{io}\n### Signature:\n{py_sig}\n{PY_CODE_TAG}"
    raise ValueError(f"unknown/unsupported task for python: {task}")


def build_example_py(task: str, rec: dict) -> dict | None:
    try:
        mod = parse(rec["program_canonical"])
        code = to_python(mod)
        ent = entry_name(mod)
        sig = py_signature(mod)
    except NotImplementedError:
        return None  # record uses features outside the Python subset; skip
    prompt = build_prompt_py(task, rec, sig)
    io_examples = rec.get("io_examples", [])
    eval_io = io_examples[PROMPT_K:] if task == "io_to_code" else io_examples
    return {
        "task": task,
        "prompt": prompt,
        "completion": code,
        "id": rec["id"],
        "ast_hash": rec["ast_hash"],
        "type_signature": sig,
        "spec_structured": rec.get("spec_structured", ""),
        "io_examples": io_examples,
        "eval_io": eval_io,
        "entry": ent,
        "reference": code,
        "lang": "python",
    }


def export_python_dataset(in_dir: str, out_dir: str, tasks: list[str] | None = None,
                          test_pct: int = 15, max_examples: int | None = None) -> dict:
    tasks = tasks or ["spec_to_code"]
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.jsonl")
    test_path = os.path.join(out_dir, "test.jsonl")

    train: list[dict] = []
    test: list[dict] = []
    for rec in iter_records(in_dir):
        is_test = _is_test(rec["ast_hash"], test_pct)
        for task in tasks:
            ex = build_example_py(task, rec)
            if ex is None:
                continue
            if is_test and task == "io_to_code" and not ex["eval_io"]:
                continue
            (test if is_test else train).append(ex)
        if max_examples and (len(train) + len(test)) >= max_examples:
            break

    train_prompts = {ex["prompt"] for ex in train}
    kept_test = [ex for ex in test if ex["prompt"] not in train_prompts]

    with open(train_path, "w") as ftr:
        for ex in train:
            ftr.write(json.dumps(ex, sort_keys=True) + "\n")
    with open(test_path, "w") as fte:
        for ex in kept_test:
            fte.write(json.dumps(ex, sort_keys=True) + "\n")

    return {
        "train": len(train),
        "test": len(kept_test),
        "test_dropped_leak": len(test) - len(kept_test),
        "train_path": train_path,
        "test_path": test_path,
    }
