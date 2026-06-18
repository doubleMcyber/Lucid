"""Export a Loom dataset into supervised (prompt, completion) pairs for LM
fine-tuning, plus a held-out test split carrying the metadata the evaluator
needs (held-out IO, signature, reference program).

Tasks:
  * spec_to_code — prompt = NL spec + signature; completion = the program.
  * io_to_code   — prompt = IO examples + signature; completion = the program.
  * completion   — prompt = a canonical prefix; completion = the remainder.

The train/test split is deterministic and leakage-free at the level the eval
actually tests — the *prompt*. The split is seeded by ast_hash (so a given
program is entirely in train or in test), and then any test prompt that also
occurs in train is dropped: distinct programs can collapse to byte-identical
prompts (e.g. two functions with the same signature + IO examples), and such a
collision would otherwise hand the model a memorizable answer across the split.

For `io_to_code`, the prompt shows only the first `PROMPT_K` IO examples while the
evaluator scores against the *held-out* remainder (`eval_io`), so exec_pass@1
measures generalization to inputs the model never saw — not reproduction of
shown examples. Test records with no held-out IO are dropped from `io_to_code`.

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

# Number of IO examples shown in an io_to_code prompt; the rest are held out for
# evaluation so exec_pass@1 measures generalization, not memorization.
PROMPT_K = 3


def _fmt_io(io_examples: list, k: int = PROMPT_K) -> str:
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
    io_examples = rec.get("io_examples", [])
    # eval_io is the IO the evaluator scores against. For io_to_code it is the
    # held-out remainder (the prompt shows only the first PROMPT_K); for other
    # tasks no IO is shown, so the full set is held out.
    eval_io = io_examples[PROMPT_K:] if task == "io_to_code" else io_examples
    return {
        "task": task,
        "prompt": prompt,
        "completion": completion,
        "id": rec["id"],
        "ast_hash": rec["ast_hash"],
        "type_signature": rec.get("type_signature", ""),
        "spec_structured": rec.get("spec_structured", ""),
        "io_examples": io_examples,
        "eval_io": eval_io,
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

    # Pass 1: build every example and assign a side by ast_hash (program-level
    # disjointness). Buffering is fine — corpora here are tens of thousands of
    # small records.
    train: list[dict] = []
    test: list[dict] = []
    for rec in iter_records(in_dir):
        is_test = _is_test(rec["ast_hash"], test_pct)
        for task in tasks:
            ex = build_example(task, rec)
            if ex is None:
                continue
            # Drop io_to_code test items with no held-out IO — exec_pass@1 on them
            # would only reward reproducing examples shown in the prompt.
            if is_test and task == "io_to_code" and not ex["eval_io"]:
                continue
            (test if is_test else train).append(ex)
        if max_examples and (len(train) + len(test)) >= max_examples:
            break

    # Pass 2: enforce prompt-level leakage-freedom. Distinct programs can collapse
    # to identical prompts; drop any test prompt that also appears in train so the
    # model is never scored on a prompt for which it saw a memorizable answer.
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
