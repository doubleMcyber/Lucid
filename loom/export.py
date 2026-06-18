"""Export a Loom dataset into supervised (prompt, completion) pairs for LM
fine-tuning, plus a held-out test split carrying the metadata the evaluator
needs (held-out IO, signature, reference program).

Tasks:
  * spec_to_code — prompt = NL spec + signature; completion = the program.
  * io_to_code   — prompt = IO examples + signature; completion = the program.
  * completion   — prompt = a canonical prefix; completion = the remainder.

The train/test split is deterministic and leakage-free: it hashes the program's
ast_hash, so a given program is always entirely in train or entirely in test.

The prompt templates live here so training and evaluation use the *identical*
format (a common source of silent eval skew).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterator

# Sentinel that separates prompt from code in every template.
CODE_TAG = "### Lucid:\n"


def _fmt_io(io_examples: list, k: int = 3) -> str:
    lines = []
    for ex in io_examples[:k]:
        lines.append(f"  {json.dumps(ex['input'])} -> {json.dumps(ex['output'])}")
    return "\n".join(lines)


def build_prompt(task: str, rec: dict) -> str:
    sig = rec.get("type_signature", "")
    if task == "spec_to_code":
        spec = rec.get("spec_structured", "")
        return (f"### Spec:\n{spec}\n### Signature:\n{sig}\n{CODE_TAG}")
    if task == "io_to_code":
        io = _fmt_io(rec.get("io_examples", []))
        return (f"### Examples (input -> output):\n{io}\n### Signature:\n{sig}\n{CODE_TAG}")
    if task == "completion":
        cps = rec.get("completion_pairs", [])
        comp = next((c for c in cps if c["kind"] == "completion"), None)
        if comp is None:
            return ""
        return comp["prefix"]
    raise ValueError(f"unknown task {task}")


def build_example(task: str, rec: dict) -> dict | None:
    prompt = build_prompt(task, rec)
    if not prompt:
        return None
    if task == "completion":
        cps = rec.get("completion_pairs", [])
        comp = next((c for c in cps if c["kind"] == "completion"), None)
        completion = comp["completion"]
    else:
        completion = rec["program_canonical"]
    return {
        "task": task,
        "prompt": prompt,
        "completion": completion,
        "id": rec["id"],
        "ast_hash": rec["ast_hash"],
        "type_signature": rec.get("type_signature", ""),
        "spec_structured": rec.get("spec_structured", ""),
        "io_examples": rec.get("io_examples", []),
        "reference": rec["program_canonical"],
    }


def iter_records(in_dir: str) -> Iterator[dict]:
    for path in sorted(glob.glob(os.path.join(in_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _is_test(ast_hash: str, test_pct: int) -> bool:
    return (int(ast_hash[:8], 16) % 100) < test_pct


def export_dataset(in_dir: str, out_dir: str, tasks: list[str] | None = None,
                   test_pct: int = 15, max_examples: int | None = None) -> dict:
    tasks = tasks or ["spec_to_code"]
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.jsonl")
    test_path = os.path.join(out_dir, "test.jsonl")
    n_train = n_test = 0
    with open(train_path, "w") as ftr, open(test_path, "w") as fte:
        for rec in iter_records(in_dir):
            is_test = _is_test(rec["ast_hash"], test_pct)
            for task in tasks:
                ex = build_example(task, rec)
                if ex is None:
                    continue
                line = json.dumps(ex, sort_keys=True) + "\n"
                if is_test:
                    fte.write(line)
                    n_test += 1
                else:
                    ftr.write(line)
                    n_train += 1
            if max_examples and (n_train + n_test) >= max_examples:
                break
    return {"train": n_train, "test": n_test, "train_path": train_path, "test_path": test_path}
