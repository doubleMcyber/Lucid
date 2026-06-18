"""Labelers — the pairing engines (PRD §8.3).

Each accepted program is turned into several kinds of training pairs. Every
labeler is a pure function of (module, validation report, rng) and reuses the one
toolchain, so labels are correct by construction:

  * IO examples        — (input, output) from the validator's successful runs.
  * Execution trace    — step-by-step trace for trace/explain training.
  * Completion / FIM   — prefix->continuation and span-infill, clean because the
                         canonical form is line-structured and unambiguous.
  * Repair             — inject a typed mutation (type-break or behavior-break);
                         pair (broken, error/diff, fixed). The break is *verified*.
  * Refactor / equiv.  — apply a semantics-preserving transform; pair (A, B) and
                         verify behavioral equivalence on the IO set.
  * Specification      — deterministic templated structured + NL description from
                         the AST; optional LLM paraphrase kept only if it passes
                         back-translation (PRD §13).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Optional

from lucid.ast import (
    BoolLit, BuiltinCall, Cond, Expr, FnDecl, IntLit, Program, StrLit,
)
from lucid.errors import RuntimeError_, TypeError_
from lucid.interp import ExecConfig, Interpreter, value_to_json
from lucid.parser import Module, parse
from lucid.printer import print_program
from lucid.typechecker import typecheck
from lucid.types import BOOL, INT, STR, TList, Type

from . import features as feat
from .astutil import (
    collect_local_names, count_exprs, rename_local_in_fn, replace_nth_expr,
)
from .rng import Rng
from .validator import ValidationReport


# ==========================================================================
# IO + trace
# ==========================================================================
def io_examples(report: ValidationReport, limit: int = 4) -> list[dict]:
    return [{"input": p.input_json, "output": p.output_json} for p in report.io[:limit]]


def execution_trace(module: Module, report: ValidationReport,
                    exec_config: Optional[ExecConfig] = None) -> Optional[dict]:
    if not report.io:
        return None
    cfg = replace(exec_config or ExecConfig(), trace=True)
    pair = report.io[0]
    try:
        interp = Interpreter(module, cfg)
        res = interp.run_entry(pair.raw_input)
    except RuntimeError_:
        return None
    return {
        "input": pair.input_json,
        "output": value_to_json(res.value),
        "steps": res.steps,
        "trace": res.trace,
    }


# ==========================================================================
# Completion / fill-in-the-middle
# ==========================================================================
def completion_pairs(canonical: str, rng: Rng) -> list[dict]:
    """Line-structured completion + infill. The canonical form puts each
    statement on its own line, so splits land on clean boundaries."""
    lines = canonical.split("\n")
    pairs: list[dict] = []
    if len(lines) >= 2:
        k = rng.randint(1, len(lines) - 1)
        pairs.append({
            "kind": "completion",
            "prefix": "\n".join(lines[:k]) + "\n",
            "completion": "\n".join(lines[k:]),
        })
        # span-infill: blank out one interior line
        j = rng.randint(0, len(lines) - 1)
        pairs.append({
            "kind": "infill",
            "prefix": "\n".join(lines[:j]) + ("\n" if j > 0 else ""),
            "suffix": ("\n" if j < len(lines) - 1 else "") + "\n".join(lines[j + 1:]),
            "middle": lines[j],
        })
    return pairs


# ==========================================================================
# Repair
# ==========================================================================
# same-signature builtin swaps that change behaviour but preserve types
_BEHAVIOR_SWAPS = {
    "add": "sub", "sub": "add", "mul": "add", "div": "mul",
    "min": "max", "max": "min",
    "lt_int": "gt_int", "gt_int": "lt_int", "le_int": "ge_int", "ge_int": "le_int",
    "eq_int": "ne_int", "ne_int": "eq_int",
    "and": "or", "or": "and", "eq_bool": "ne_bool", "ne_bool": "eq_bool",
    "eq_str": "ne_str", "ne_str": "eq_str",
}


def _run_all(module: Module, report: ValidationReport, cfg: ExecConfig) -> Optional[list]:
    outs = []
    for p in report.io:
        try:
            outs.append(value_to_json(Interpreter(module, cfg).run_entry(p.raw_input).value))
        except RuntimeError_:
            outs.append("<error>")
    return outs


def repair_pairs(module: Module, fixed_canonical: str, report: ValidationReport,
                 rng: Rng, exec_config: Optional[ExecConfig] = None) -> list[dict]:
    cfg = exec_config or ExecConfig()
    pairs: list[dict] = []
    prog = module.program

    # --- behavioral break: swap a builtin to a same-signature sibling -------
    swappable = count_exprs(prog, lambda e: isinstance(e, BuiltinCall) and e.name in _BEHAVIOR_SWAPS)
    if swappable:
        base_out = _run_all(module, report, cfg)
        idx = rng.randint(0, swappable - 1)

        def swap(e: Expr) -> Expr:
            return replace(e, name=_BEHAVIOR_SWAPS[e.name])

        broken = replace_nth_expr(prog, lambda e: isinstance(e, BuiltinCall) and e.name in _BEHAVIOR_SWAPS, idx, swap)
        bmod = Module(broken, module.tenv)
        try:
            typecheck(bmod)  # behavioral break still typechecks
            bout = _run_all(bmod, report, cfg)
            diff_idx = next((i for i, (a, b) in enumerate(zip(base_out, bout)) if a != b), None)
            if diff_idx is not None:
                pairs.append({
                    "kind": "repair_behavior",
                    "broken": print_program(broken),
                    "fixed": fixed_canonical,
                    "diff_input": report.io[diff_idx].input_json,
                    "broken_output": bout[diff_idx],
                    "fixed_output": base_out[diff_idx],
                })
        except TypeError_:
            pass

    # --- type break: corrupt a literal to the wrong type --------------------
    pair = _type_break(prog, fixed_canonical, rng)
    if pair is not None:
        pairs.append(pair)

    return pairs


def _type_break(prog: Program, fixed_canonical: str, rng: Rng) -> Optional[dict]:
    """Replace an int literal with a bool literal (or vice versa). In a strongly
    typed, coercion-free language this almost always breaks the type check; we
    keep the pair only if it genuinely does, and capture the error message."""
    n_int = count_exprs(prog, lambda e: isinstance(e, IntLit))
    n_bool = count_exprs(prog, lambda e: isinstance(e, BoolLit))
    plans = []
    if n_int:
        plans.append(("int", n_int))
    if n_bool:
        plans.append(("bool", n_bool))
    if not plans:
        return None
    kind, n = rng.choice(plans)
    idx = rng.randint(0, n - 1)
    if kind == "int":
        broken = replace_nth_expr(prog, lambda e: isinstance(e, IntLit), idx, lambda e: BoolLit(True))
    else:
        broken = replace_nth_expr(prog, lambda e: isinstance(e, BoolLit), idx, lambda e: IntLit(0))
    try:
        typecheck(Module(broken, _tenv_of(prog)))
        return None  # didn't break — skip
    except TypeError_ as ex:
        return {
            "kind": "repair_type",
            "broken": print_program(broken),
            "fixed": fixed_canonical,
            "error": str(ex),
        }


def _tenv_of(prog: Program):
    # Rebuild a TypeEnv by re-parsing the canonical text (cheap, robust).
    return parse(print_program(prog)).tenv


# ==========================================================================
# Refactor / equivalence
# ==========================================================================
def refactor_pairs(module: Module, original_canonical: str, report: ValidationReport,
                   rng: Rng, exec_config: Optional[ExecConfig] = None) -> list[dict]:
    cfg = exec_config or ExecConfig()
    prog = module.program
    transforms: list[tuple[str, Program]] = []

    # 1. rename a local variable in the entry function
    entry_idx = len(prog.decls) - 1
    entry = prog.decls[entry_idx]
    if isinstance(entry, FnDecl):
        locals_ = collect_local_names(entry)
        if locals_:
            old = rng.choice(locals_)
            new = f"r_{old}"
            renamed = rename_local_in_fn(entry, old, new)
            transforms.append(("rename", Program(prog.decls[:entry_idx] + (renamed,))))

    # 2. identity-wrap a random sub-expression with `cond true then e else e`
    n = count_exprs(prog, lambda e: True)
    if n:
        idx = rng.randint(0, n - 1)
        wrapped = replace_nth_expr(prog, lambda e: True, idx,
                                   lambda e: Cond(BoolLit(True), e, e))
        transforms.append(("cond_identity", wrapped))

    pairs: list[dict] = []
    base_out = _run_all(module, report, cfg)
    for name, newprog in transforms:
        nmod = Module(newprog, module.tenv)
        try:
            typecheck(nmod)
        except TypeError_:
            continue
        out = _run_all(nmod, report, cfg)
        if out == base_out:  # verified behaviorally equivalent
            pairs.append({
                "kind": "refactor",
                "transform": name,
                "a": original_canonical,
                "b": print_program(newprog),
                "relation": "equivalent",
            })
    return pairs


# ==========================================================================
# Specification (templated structured + NL; optional validated paraphrase)
# ==========================================================================
_TYPE_WORDS = {
    "#Int": "an integer", "#Bool": "a boolean", "#Str": "a string",
}


def _describe_type(t: Type) -> str:
    if isinstance(t, TList):
        return f"a list of {_describe_type_plural(t.elem)}"
    s = str(t)
    if s in _TYPE_WORDS:
        return _TYPE_WORDS[s]
    return f"a value of type {s}"


def _describe_type_plural(t: Type) -> str:
    if t == INT:
        return "integers"
    if t == BOOL:
        return "booleans"
    if t == STR:
        return "strings"
    if isinstance(t, TList):
        return f"lists of {_describe_type_plural(t.elem)}"
    return f"values of type {t}"


def spec_for(module: Module) -> dict:
    entry = module.program.entry()
    params = entry.params
    feats = sorted(feat.features(module.program))

    sig = "({}) -> {}".format(
        ", ".join(str(t) for _, t in params), str(entry.ret)
    )
    if params:
        param_desc = ", ".join(_describe_type(t) for _, t in params)
        nl = (f"Define a function @{entry.name} that takes {param_desc} "
              f"and returns {_describe_type(entry.ret)}.")
    else:
        nl = (f"Define a function @{entry.name} that takes no inputs "
              f"and returns {_describe_type(entry.ret)}.")
    ops = [f.replace("builtin_", "") for f in feats if f.startswith("builtin_")]
    if ops:
        nl += " It uses: " + ", ".join(sorted(ops)[:8]) + "."

    return {
        "type_signature": sig,
        "spec_structured": nl,
        "features": feats,
    }


# A paraphraser is (structured_spec_text) -> nl_paraphrase or None.
Paraphraser = Callable[[str], Optional[str]]
# A synthesizer is (nl_paraphrase) -> Lucid source or None (the back-translator).
Synthesizer = Callable[[str], Optional[str]]


def validated_paraphrase(spec: dict, module: Module, report: ValidationReport,
                         paraphraser: Optional[Paraphraser],
                         synthesizer: Optional[Synthesizer],
                         exec_config: Optional[ExecConfig] = None) -> Optional[dict]:
    """Back-translation validation (PRD §13): keep an LLM paraphrase only if
    regenerating code from it reproduces the original program's IO behavior.

    Both `paraphraser` and `synthesizer` are pluggable (any LLM, or a mock). If
    either is absent, no paraphrase is produced — the templated spec stands alone.
    """
    if paraphraser is None or synthesizer is None:
        return None
    para = paraphraser(spec["spec_structured"])
    if not para:
        return None
    src = synthesizer(para)
    if not src:
        return {"spec_nl_paraphrase": para, "spec_verified": False}
    try:
        cand = parse(src)
        typecheck(cand)
    except Exception:
        return {"spec_nl_paraphrase": para, "spec_verified": False}
    cfg = exec_config or ExecConfig()
    # behavioral equivalence on the IO set
    for p in report.io:
        try:
            got = value_to_json(Interpreter(cand, cfg).run_entry(p.raw_input).value)
        except RuntimeError_:
            return {"spec_nl_paraphrase": para, "spec_verified": False}
        if got != p.output_json:
            return {"spec_nl_paraphrase": para, "spec_verified": False}
    return {"spec_nl_paraphrase": para, "spec_verified": True}
