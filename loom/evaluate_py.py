"""Model-agnostic evaluation harness for the Python baseline (mirror of
`loom.evaluate`, but Python has no static type system and is not total).

Metrics:
  * parse_rate   — fraction whose generation parses as Python AND defines a
                   function (an empty/function-less snippet is not a program).
  * runs_rate    — fraction that executes on all held-out inputs without raising
                   (the closest *dynamic* analog to Lucid's static typecheck;
                   reported, with the asymmetry noted, not as an equal metric).
  * exec_pass@1  — fraction that runs and reproduces every held-out output (H2).
  * exact_match  — fraction whose stripped source equals the reference.

Execution is sandboxed (`loom.sandbox`): separate process, wall-clock timeout,
restricted builtins — because generated Python, unlike Lucid, can loop forever or
touch the system.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Optional

from loom.sandbox import TIMEOUT_DEFAULT, run_one

_STOP_MARKERS = ["\n### ", "\n###", "```", "<|endoftext|>", "<|im_end|>", "\n\n\n"]


def extract_program_py(text: str) -> str:
    s = text
    for m in _STOP_MARKERS:
        idx = s.find(m)
        if idx != -1:
            s = s[:idx]
    s = s.strip()
    try:
        ast.parse(s)
        return s
    except SyntaxError:
        pass
    lines = s.split("\n")
    for end in range(len(lines) - 1, 0, -1):
        if len(lines) - end > 200:
            break
        cand = "\n".join(lines[:end]).strip()
        if not cand:
            continue
        try:
            ast.parse(cand)
            return cand
        except SyntaxError:
            continue
    return s


def _functions(tree: ast.Module) -> list[str]:
    return [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]


@dataclass
class EvalResultPy:
    n: int = 0
    n_parse: int = 0
    n_runs: int = 0
    n_exec_pass: int = 0
    n_exact: int = 0
    details: list = field(default_factory=list)

    @property
    def parse_rate(self) -> float:
        return self.n_parse / self.n if self.n else 0.0

    @property
    def runs_rate(self) -> float:
        return self.n_runs / self.n if self.n else 0.0

    @property
    def exec_pass_at_1(self) -> float:
        return self.n_exec_pass / self.n if self.n else 0.0

    @property
    def exact_match(self) -> float:
        return self.n_exact / self.n if self.n else 0.0

    def summary(self) -> dict:
        return {
            "n": self.n,
            "parse_rate": round(self.parse_rate, 4),
            "runs_rate": round(self.runs_rate, 4),
            "exec_pass@1": round(self.exec_pass_at_1, 4),
            "exact_match": round(self.exact_match, 4),
        }


def _matches(got_outputs: list, eval_io: list) -> bool:
    if not eval_io or len(got_outputs) != len(eval_io):
        return False
    for o, ex in zip(got_outputs, eval_io):
        if not o.get("ok"):
            return False
        if json.dumps(o["output"], sort_keys=True) != json.dumps(ex["output"], sort_keys=True):
            return False
    return True


def evaluate_py(items: list[tuple[dict, str]], timeout: float = TIMEOUT_DEFAULT,
                keep_details: bool = False) -> EvalResultPy:
    res = EvalResultPy()
    for rec, gen in items:
        res.n += 1
        code = extract_program_py(gen)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            if keep_details:
                res.details.append({"id": rec.get("id"), "stage": "parse_fail"})
            continue
        fns = _functions(tree)
        if not code.strip() or not fns:
            if keep_details:
                res.details.append({"id": rec.get("id"), "stage": "empty"})
            continue
        res.n_parse += 1
        expected = rec.get("entry", "main")
        entry = expected if expected in fns else fns[-1]
        eval_io = rec.get("eval_io", rec.get("io_examples", []))
        inputs = [ex["input"] for ex in eval_io]
        out = run_one(code, entry, inputs, timeout=timeout) if inputs else {"status": "noio"}
        if out.get("status") == "ok":
            outputs = out["outputs"]
            if inputs and all(o.get("ok") for o in outputs) and len(outputs) == len(inputs):
                res.n_runs += 1
            if _matches(outputs, eval_io):
                res.n_exec_pass += 1
        if code.strip() == (rec.get("reference", "\0") or "\0").strip():
            res.n_exact += 1
        if keep_details:
            res.details.append({"id": rec.get("id"), "status": out.get("status")})
    return res
