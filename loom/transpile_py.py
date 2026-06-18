"""Lucid → Python transpiler — the *matched baseline* for the language thesis.

To test H1–H4 ("does model-native syntax help?") we need a mainstream-language
corpus drawn from the *identical task distribution* as the Lucid corpus. Rather
than sample Python independently (which would confound task difficulty with
language), we transpile each Loom-generated Lucid program to behaviorally
equivalent Python. Same tasks, same IO, same train/test split — only the surface
language differs.

Faithfulness: Lucid is total, so the Python we emit reproduces Lucid's *total*
semantics inline (e.g. `div`-by-zero is 0, not an exception; `get_or`/`head_or`
bounds-check) — otherwise the Python "reference" would diverge from the held-out
IO and the baseline would not be matched. The output is clean, self-contained
Python (no helper preamble) so a model learns idiomatic Python, not a runtime.

Coverage matches the experiment's `tiny_config` and a bit more: ints, bools,
strings, lists, arithmetic/comparison/boolean builtins, list intrinsics,
higher-order `map`/`filter`/`foldl`, `foreach`, `if`, `cond`. Records, variants,
and `match` raise NotImplementedError (the matched experiment excludes them).
"""

from __future__ import annotations

import keyword

from lucid.ast import (BoolLit, BuiltinCall, Call, Cond, Field, Foreach, FnDecl,
                       FnRef, IfStmt, IntLit, Let, ListLit, Match, RecordLit,
                       Return, SetStmt, StrLit, VarDecl, VarRef, VariantLit)
from lucid.parser import Module
from lucid.types import TBool, TInt, TList, TStr

# Python names the transpiler emits as bare calls; a Lucid identifier equal to
# any of these (or a Python keyword) is renamed so it can never shadow them.
_RESERVED = set(keyword.kwlist) | {
    "len", "abs", "min", "max", "range", "list", "reversed", "str", "int",
    "bool", "map", "filter", "functools", "reduce", "True", "False", "None",
}


def _pyname(n: str) -> str:
    return f"lc_{n}" if n in _RESERVED else n


# ---- binary/unary builtin templates (operands already parenthesized) --------
_BIN = {
    "add": "{0} + {1}", "sub": "{0} - {1}", "mul": "{0} * {1}",
    "min": "min({0}, {1})", "max": "max({0}, {1})",
    "lt_int": "{0} < {1}", "le_int": "{0} <= {1}", "gt_int": "{0} > {1}",
    "ge_int": "{0} >= {1}", "eq_int": "{0} == {1}", "ne_int": "{0} != {1}",
    "and": "{0} and {1}", "or": "{0} or {1}", "xor": "{0} != {1}",
    "eq_bool": "{0} == {1}", "ne_bool": "{0} != {1}",
    "concat": "{0} + {1}", "eq_str": "{0} == {1}", "ne_str": "{0} != {1}",
    "append": "{0} + [{1}]", "concat_list": "{0} + {1}",
}
_UN = {
    "neg": "-{0}", "abs": "abs({0})", "not": "not {0}",
    "len_str": "len({0})", "to_str_int": "str({0})",
    "length": "len({0})", "is_empty": "len({0}) == 0",
    "reverse": "list(reversed({0}))",
}


class _NeedsFunctools(Exception):
    pass


def _expr(e) -> str:
    """Render a Lucid expression as a *parenthesized* Python expression."""
    if isinstance(e, IntLit):
        return str(e.value)
    if isinstance(e, BoolLit):
        return "True" if e.value else "False"
    if isinstance(e, StrLit):
        return repr(e.value)
    if isinstance(e, VarRef):
        return _pyname(e.name)
    if isinstance(e, FnRef):
        return _pyname(e.name)
    if isinstance(e, Call):
        return f"{_pyname(e.name)}(" + ", ".join(_expr(a) for a in e.args) + ")"
    if isinstance(e, ListLit):
        return "[" + ", ".join(_expr(x) for x in e.elems) + "]"
    if isinstance(e, Cond):
        return f"({_expr(e.then)} if {_expr(e.test)} else {_expr(e.els)})"
    if isinstance(e, BuiltinCall):
        return "(" + _builtin(e) + ")"
    if isinstance(e, (RecordLit, Field, VariantLit, Match)):
        raise NotImplementedError(f"transpile: {type(e).__name__}")
    raise NotImplementedError(f"transpile: {type(e).__name__}")


def _builtin(e: BuiltinCall) -> str:
    a = [_expr(x) for x in e.args]
    name = e.name
    if name in _BIN:
        return _BIN[name].format(f"({a[0]})", f"({a[1]})")
    if name in _UN:
        return _UN[name].format(f"({a[0]})")
    if name == "to_str_bool":
        return f'("true" if ({a[0]}) else "false")'
    if name == "div":
        return f"(({a[0]}) // ({a[1]}) if ({a[1]}) != 0 else 0)"
    if name == "mod":
        return f"(({a[0]}) % ({a[1]}) if ({a[1]}) != 0 else 0)"
    if name == "range":
        return f"list(range(({a[0]}), ({a[1]})))"
    if name == "get_or":
        return f"(({a[0]})[({a[1]})] if 0 <= ({a[1]}) < len({a[0]}) else ({a[2]}))"
    if name == "head_or":
        return f"(({a[0]})[0] if ({a[0]}) else ({a[1]}))"
    if name == "map":
        return f"[{a[1]}(_x) for _x in ({a[0]})]"
    if name == "filter":
        return f"[_x for _x in ({a[0]}) if {a[1]}(_x)]"
    if name == "apply1":
        return f"{a[0]}({a[1]})"
    if name == "apply2":
        return f"{a[0]}({a[1]}, {a[2]})"
    if name == "foldl":
        raise _NeedsFunctools()
    raise NotImplementedError(f"transpile builtin: {name}")


def _foldl(e: BuiltinCall) -> str:
    a = [_expr(x) for x in e.args]
    return f"functools.reduce(lambda _acc, _x: {a[2]}(_acc, _x), ({a[0]}), ({a[1]}))"


def _stmts(body, indent: int, ctx: dict) -> list[str]:
    pad = "    " * indent
    out: list[str] = []
    for s in body:
        if isinstance(s, (Let, VarDecl)):
            out.append(f"{pad}{_pyname(s.name)} = {_rhs(s.expr, ctx)}")
        elif isinstance(s, SetStmt):
            out.append(f"{pad}{_pyname(s.name)} = {_rhs(s.expr, ctx)}")
        elif isinstance(s, Return):
            out.append(f"{pad}return {_rhs(s.expr, ctx)}")
        elif isinstance(s, Foreach):
            out.append(f"{pad}for {_pyname(s.var)} in {_rhs(s.iter, ctx)}:")
            out += _stmts(s.body, indent + 1, ctx) or [f"{pad}    pass"]
        elif isinstance(s, IfStmt):
            out.append(f"{pad}if {_rhs(s.cond, ctx)}:")
            out += _stmts(s.then_body, indent + 1, ctx) or [f"{pad}    pass"]
            if s.else_body:
                out.append(f"{pad}else:")
                out += _stmts(s.else_body, indent + 1, ctx)
        else:
            raise NotImplementedError(f"transpile stmt: {type(s).__name__}")
    return out


def _rhs(e, ctx: dict) -> str:
    """Top-level expression render that records foldl's functools need."""
    try:
        return _expr(e)
    except _NeedsFunctools:
        ctx["functools"] = True
        # re-render with foldl support by walking once more is overkill; instead
        # handle the common case where the whole rhs is a foldl call.
        if isinstance(e, BuiltinCall) and e.name == "foldl":
            return _foldl(e)
        raise NotImplementedError("nested foldl not supported")


def _ty_py(t) -> str:
    if isinstance(t, TInt):
        return "int"
    if isinstance(t, TBool):
        return "bool"
    if isinstance(t, TStr):
        return "str"
    if isinstance(t, TList):
        return f"list[{_ty_py(t.elem)}]"
    raise NotImplementedError(f"transpile type: {t}")


def _fn(decl: FnDecl, ctx: dict) -> str:
    params = ", ".join(f"{_pyname(n)}: {_ty_py(t)}" for n, t in decl.params)
    head = f"def {_pyname(decl.name)}({params}) -> {_ty_py(decl.ret)}:"
    body = _stmts(decl.body, 1, ctx) or ["    pass"]
    return head + "\n" + "\n".join(body)


def to_python(module: Module) -> str:
    """Full Python source for a Lucid module (all functions, entry last)."""
    ctx: dict = {}
    fns = [_fn(d, ctx) for d in module.program.functions()]
    preamble = "import functools\n\n" if ctx.get("functools") else ""
    return preamble + "\n\n".join(fns) + "\n"


def entry_name(module: Module) -> str:
    return _pyname(module.program.entry().name)


def py_signature(module: Module) -> str:
    """Human-facing Python signature of the entry, for prompts."""
    fn = module.program.entry()
    params = ", ".join(_ty_py(t) for _, t in fn.params)
    return f"({params}) -> {_ty_py(fn.ret)}"
