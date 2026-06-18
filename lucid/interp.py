"""Total interpreter and step tracer for Lucid (PRD §7.5).

A Lucid program is a pure function from inputs to an output: no ambient I/O, no
nondeterminism. Termination is guaranteed by construction (DAG call graph +
finite `foreach`), so the executor-in-the-loop always finishes — the property
that makes generating IO pairs at scale practical.

`ExecConfig` adds *resource bounds* on top of guaranteed termination: caps on
total work, integer magnitude, string/list size, and call depth. A program that
trips a bound raises `ResourceError` and is simply discarded by the validator
(expected, not a bug). These bounds keep execution fast and IO values realistic.

The tracer, when enabled, records a structured step list for trace/explain
training pairs. Values are serialized with `value_to_json` (JSON-ready, with a
namespaced encoding for records/variants/function values).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .ast import (
    BoolLit, BuiltinCall, Call, Cond, Expr, Field, FnDecl, FnRef, Foreach,
    IfStmt, IntLit, Let, ListLit, Match, Program, RecordLit, Return, SetStmt,
    Stmt, StrLit, VarDecl, VarRef, VariantLit,
)
from .builtins_def import BUILTINS
from .errors import ResourceError, RuntimeError_
from .parser import Module
from .values import FnVal, RecordVal, VariantVal


@dataclass
class ExecConfig:
    max_work: int = 200_000      # total charged steps before discard
    max_list: int = 2_000        # max length of any list value
    max_str_len: int = 4_096     # max length of any string value
    max_int_abs: int = 10 ** 12  # max magnitude of any integer value
    max_depth: int = 128         # max call-stack depth (DAG, so finite anyway)
    trace: bool = False
    max_trace: int = 10_000


@dataclass
class RunResult:
    value: Any
    steps: int
    trace: list = field(default_factory=list)


class _Return:
    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value


def value_to_json(v: Any) -> Any:
    """Convert a runtime value to a JSON-serializable structure."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, str)):
        return v
    if isinstance(v, list):
        return [value_to_json(x) for x in v]
    if isinstance(v, RecordVal):
        return {"@record": v.type_name, "@fields": {n: value_to_json(x) for n, x in v.fields}}
    if isinstance(v, VariantVal):
        return {"@variant": v.type_name, "@tag": v.tag, "@args": [value_to_json(x) for x in v.args]}
    if isinstance(v, FnVal):
        return {"@fn": v.name}
    raise TypeError(f"cannot serialize value {v!r}")


def value_from_json(j: Any) -> Any:
    """Inverse of `value_to_json`: reconstruct a runtime value from its JSON
    form. The record/variant/function encodings are self-describing, so no type
    information is needed. Used by the evaluator to feed held-out IO inputs into
    a model-generated program."""
    if isinstance(j, bool):
        return j
    if isinstance(j, (int, str)):
        return j
    if isinstance(j, list):
        return [value_from_json(x) for x in j]
    if isinstance(j, dict):
        if "@record" in j:
            fields = j["@fields"]
            return RecordVal(j["@record"], tuple((k, value_from_json(v)) for k, v in fields.items()))
        if "@variant" in j:
            return VariantVal(j["@variant"], j["@tag"], tuple(value_from_json(x) for x in j["@args"]))
        if "@fn" in j:
            return FnVal(j["@fn"])
    raise ValueError(f"cannot deserialize value {j!r}")


class Interpreter:
    def __init__(self, module: Module, config: Optional[ExecConfig] = None):
        self.program: Program = module.program
        self.fns: dict[str, FnDecl] = {fn.name: fn for fn in module.program.functions()}
        self.config = config or ExecConfig()
        self.max_list = self.config.max_list
        self.work = 0
        self.depth = 0
        self.trace_events: list = []

    # -- resource budget (referenced by built-ins as ctx.charge) -----------
    def charge(self, n: int) -> None:
        self.work += n
        if self.work > self.config.max_work:
            raise ResourceError("work budget exceeded")

    def _guard(self, v: Any) -> Any:
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            if abs(v) > self.config.max_int_abs:
                raise ResourceError("integer magnitude bound exceeded")
        elif isinstance(v, str):
            if len(v) > self.config.max_str_len:
                raise ResourceError("string length bound exceeded")
        elif isinstance(v, list):
            if len(v) > self.max_list:
                raise ResourceError("list length bound exceeded")
        return v

    def _emit(self, ev: dict) -> None:
        if self.config.trace and len(self.trace_events) < self.config.max_trace:
            self.trace_events.append(ev)

    # -- public entry ------------------------------------------------------
    def run_entry(self, inputs: list) -> RunResult:
        entry = self.program.entry()
        if len(inputs) != len(entry.params):
            raise RuntimeError_(
                f"entry @{entry.name} expects {len(entry.params)} inputs"
            )
        val = self.call(entry.name, list(inputs))
        return RunResult(val, self.work, list(self.trace_events))

    # -- function application (also used by higher-order built-ins) --------
    def apply(self, fnval: FnVal, args: list) -> Any:
        return self.call(fnval.name, args)

    def call(self, name: str, args: list) -> Any:
        self.charge(1)
        self.depth += 1
        if self.depth > self.config.max_depth:
            raise ResourceError("call depth bound exceeded")
        fn = self.fns[name]
        env: dict[str, Any] = {p[0]: a for p, a in zip(fn.params, args)}
        self._emit({"event": "call", "fn": name,
                    "args": [value_to_json(a) for a in args]})
        r = self.exec_block(fn.body, env)
        self.depth -= 1
        if r is None:  # unreachable if type-checked, but stay total
            raise RuntimeError_(f"@{name} finished without returning")
        self._emit({"event": "return", "fn": name, "value": value_to_json(r.value)})
        return r.value

    # -- statements --------------------------------------------------------
    def exec_block(self, stmts: tuple[Stmt, ...], env: dict) -> Optional[_Return]:
        for s in stmts:
            r = self.exec_stmt(s, env)
            if r is not None:
                return r
        return None

    def exec_stmt(self, s: Stmt, env: dict) -> Optional[_Return]:
        self.charge(1)
        if isinstance(s, (Let, VarDecl)):
            v = self.eval(s.expr, env)
            env[s.name] = v
            self._emit({"event": "bind", "name": s.name, "value": value_to_json(v)})
            return None
        if isinstance(s, SetStmt):
            v = self.eval(s.expr, env)
            env[s.name] = v
            self._emit({"event": "set", "name": s.name, "value": value_to_json(v)})
            return None
        if isinstance(s, Foreach):
            seq = self.eval(s.iter, env)
            for x in seq:
                self.charge(1)
                env[s.var] = x
                self._emit({"event": "iter", "var": s.var, "value": value_to_json(x)})
                r = self.exec_block(s.body, env)
                if r is not None:
                    return r
            return None
        if isinstance(s, IfStmt):
            c = self.eval(s.cond, env)
            self._emit({"event": "if", "cond": bool(c)})
            branch = s.then_body if c else s.else_body
            return self.exec_block(branch, env)
        if isinstance(s, Return):
            return _Return(self.eval(s.expr, env))
        raise RuntimeError_(f"unknown statement {type(s).__name__}")

    # -- expressions -------------------------------------------------------
    def eval(self, e: Expr, env: dict) -> Any:
        self.charge(1)
        if isinstance(e, IntLit):
            return e.value
        if isinstance(e, BoolLit):
            return e.value
        if isinstance(e, StrLit):
            return e.value
        if isinstance(e, VarRef):
            return env[e.name]
        if isinstance(e, FnRef):
            return FnVal(e.name)
        if isinstance(e, BuiltinCall):
            b = BUILTINS[e.name]
            argv = [self.eval(a, env) for a in e.args]
            return self._guard(b.fn(argv, self))
        if isinstance(e, Call):
            argv = [self.eval(a, env) for a in e.args]
            return self.call(e.name, argv)
        if isinstance(e, ListLit):
            return self._guard([self.eval(el, env) for el in e.elems])
        if isinstance(e, RecordLit):
            return RecordVal(
                e.type_name,
                tuple((n, self.eval(fe, env)) for n, fe in e.fields),
            )
        if isinstance(e, Field):
            rec = self.eval(e.record, env)
            return rec.get(e.field)
        if isinstance(e, VariantLit):
            return VariantVal(
                e.type_name, e.tag, tuple(self.eval(a, env) for a in e.args)
            )
        if isinstance(e, Match):
            scrut = self.eval(e.scrutinee, env)
            for arm in e.arms:
                if arm.tag == scrut.tag:
                    for (bname, _bty), val in zip(arm.binders, scrut.args):
                        env[bname] = val
                    return self.eval(arm.body, env)
            raise RuntimeError_(f"no match arm for tag {scrut.tag}")
        if isinstance(e, Cond):
            if self.eval(e.test, env):
                return self.eval(e.then, env)
            return self.eval(e.els, env)
        raise RuntimeError_(f"unknown expression {type(e).__name__}")


def run(module: Module, inputs: list, config: Optional[ExecConfig] = None) -> RunResult:
    return Interpreter(module, config).run_entry(inputs)
