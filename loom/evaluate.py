"""Model-agnostic evaluation harness (PRD §4 H1/H2).

Given test items (each a Loom export record) and the model's generated Lucid for
each, compute:
  * parse_rate      — fraction whose generation parses (well-formedness),
  * typecheck_rate  — fraction that parses AND type-checks (H1),
  * exec_pass@1     — fraction that, run on the held-out IO inputs, reproduces
                      every expected output (H2, executed functional correctness),
  * exact_match     — fraction whose canonical form equals the reference.

`extract_program` salvages a parseable program from possibly-noisy model output
(stop markers, trailing partial lines), so a fair score is given to messy
baselines without hand-holding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from lucid.errors import LucidError, RuntimeError_
from lucid.interp import ExecConfig, Interpreter, value_from_json, value_to_json
from lucid.parser import parse
from lucid.printer import print_program
from lucid.typechecker import typecheck

_STOP_MARKERS = ["\n### ", "\n###", "```", "<|endoftext|>", "<|im_end|>", "\n\n\n"]


def extract_program(text: str) -> str:
    """Best-effort extraction of a Lucid program from raw model output."""
    s = text
    for m in _STOP_MARKERS:
        idx = s.find(m)
        if idx != -1:
            s = s[:idx]
    s = s.strip()
    # Fast path: if the whole thing already parses, return it (covers every valid
    # program with no truncation).
    try:
        parse(s)
        return s
    except LucidError:
        pass
    # Salvage: progressively drop trailing lines, but bound the number of attempts
    # so a pathological long generation can't make this an O(n^2) blow-up.
    lines = s.split("\n")
    for end in range(len(lines) - 1, 0, -1):
        if len(lines) - end > 200:
            break
        cand = "\n".join(lines[:end]).strip()
        if not cand:
            continue
        try:
            parse(cand)
            return cand
        except LucidError:
            continue
    return s


@dataclass
class EvalResult:
    n: int = 0
    n_parse: int = 0
    n_typecheck: int = 0
    n_exec_pass: int = 0
    n_exact: int = 0
    details: list = field(default_factory=list)

    @property
    def parse_rate(self) -> float:
        return self.n_parse / self.n if self.n else 0.0

    @property
    def typecheck_rate(self) -> float:
        return self.n_typecheck / self.n if self.n else 0.0

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
            "typecheck_rate": round(self.typecheck_rate, 4),
            "exec_pass@1": round(self.exec_pass_at_1, 4),
            "exact_match": round(self.exact_match, 4),
        }


def _exec_passes(module, io_examples, cfg: ExecConfig) -> bool:
    if not io_examples:
        return False
    for ex in io_examples:
        try:
            inputs = [value_from_json(v) for v in ex["input"]]
            got = value_to_json(Interpreter(module, cfg).run_entry(inputs).value)
        except (RuntimeError_, Exception):
            return False
        if got != ex["output"]:
            return False
    return True


def evaluate(items: list[tuple[dict, str]], exec_config: Optional[ExecConfig] = None,
             keep_details: bool = False) -> EvalResult:
    """items: list of (test_record, generated_text)."""
    cfg = exec_config or ExecConfig()
    res = EvalResult()
    for rec, gen in items:
        res.n += 1
        prog_text = extract_program(gen)
        parsed = None
        try:
            module = parse(prog_text)
            parsed = module
            res.n_parse += 1
        except LucidError:
            if keep_details:
                res.details.append({"id": rec.get("id"), "stage": "parse_fail"})
            continue
        typed = False
        try:
            typecheck(parsed)
            res.n_typecheck += 1
            typed = True
        except LucidError:
            pass
        if typed and _exec_passes(parsed, rec.get("io_examples", []), cfg):
            res.n_exec_pass += 1
        try:
            if print_program(parsed.program) == rec.get("reference", "\0"):
                res.n_exact += 1
        except Exception:
            pass
        if keep_details:
            res.details.append({"id": rec.get("id"), "typecheck": typed})
    return res
