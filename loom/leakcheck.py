"""Train/test leakage auditor for exported SFT datasets.

A model's score is only honest if the test set shares no *memorizable* signal with
train. There are several distinct leakage vectors, and the program-level guarantee
(`ast_hash` disjoint) does NOT imply the others:

  1. program   — a program (ast_hash) appears on both sides.
  2. prompt     — a model *input* appears on both sides (distinct programs can
                  collapse to identical prompts → a directly memorizable answer).
  3. reference  — a target program (completion) appears on both sides.
  4. shown_io   — an io_to_code test item is scored (eval_io) on an input that was
                  also shown in its prompt → reproduction, not generalization.
  5. empty      — a degenerate test item (no prompt or no held-out eval signal)
                  that can be "passed" trivially.

`audit_split` returns a count for each; a clean split has all zeros. `assert_clean`
raises with a readable diff if not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from loom.export import PROMPT_K


def _read(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


@dataclass
class LeakReport:
    n_train: int = 0
    n_test: int = 0
    program: int = 0           # ast_hash on both sides
    prompt: int = 0            # prompt string on both sides
    reference: int = 0         # reference program on both sides
    shown_io: int = 0          # io_to_code test item scored on a shown input
    empty: int = 0             # test item with no prompt or no eval signal
    examples: dict = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return (self.program == 0 and self.prompt == 0 and self.reference == 0
                and self.shown_io == 0 and self.empty == 0)

    def summary(self) -> dict:
        return {"n_train": self.n_train, "n_test": self.n_test,
                "program": self.program, "prompt": self.prompt,
                "reference": self.reference, "shown_io": self.shown_io,
                "empty": self.empty, "clean": self.clean}


def _shown_inputs(rec: dict) -> list:
    """Inputs visible in the prompt (io_to_code shows the first PROMPT_K)."""
    if rec.get("task") != "io_to_code":
        return []
    return [ex["input"] for ex in rec.get("io_examples", [])[:PROMPT_K]]


def audit_split(train_path: str, test_path: str) -> LeakReport:
    train = _read(train_path)
    test = _read(test_path)
    rep = LeakReport(n_train=len(train), n_test=len(test))

    train_progs = {e["ast_hash"] for e in train}
    train_prompts = {e["prompt"] for e in train}
    train_refs = {e["reference"] for e in train if e.get("reference")}

    prog_leak, prompt_leak, ref_leak = set(), set(), set()
    for e in test:
        if e["ast_hash"] in train_progs:
            prog_leak.add(e["ast_hash"])
        if e["prompt"] in train_prompts:
            prompt_leak.add(e["prompt"])
        if e.get("reference") and e["reference"] in train_refs:
            ref_leak.add(e["reference"])
        # held-out IO must not include any input shown in the prompt
        shown = [json.dumps(i, sort_keys=True) for i in _shown_inputs(e)]
        eval_io = e.get("eval_io", e.get("io_examples", []))
        if any(json.dumps(ex["input"], sort_keys=True) in shown for ex in eval_io):
            rep.shown_io += 1
        if not e.get("prompt", "").strip() or (e.get("task") != "completion" and not eval_io):
            rep.empty += 1

    rep.program = len(prog_leak)
    rep.prompt = len(prompt_leak)
    rep.reference = len(ref_leak)
    rep.examples = {
        "program": list(prog_leak)[:3],
        "prompt": list(prompt_leak)[:1],
        "reference": list(ref_leak)[:1],
    }
    return rep


def assert_clean(train_path: str, test_path: str) -> LeakReport:
    rep = audit_split(train_path, test_path)
    if not rep.clean:
        raise AssertionError(f"dataset leaks: {rep.summary()}  examples={rep.examples}")
    return rep


if __name__ == "__main__":
    import sys
    tr = sys.argv[1] if len(sys.argv) > 1 else "experiment/data/train.jsonl"
    te = sys.argv[2] if len(sys.argv) > 2 else "experiment/data/test.jsonl"
    r = audit_split(tr, te)
    print(json.dumps(r.summary(), indent=2))
    print("CLEAN" if r.clean else "LEAKS FOUND")
