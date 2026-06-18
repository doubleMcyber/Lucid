"""Lucid built-in operations.

All operations in Lucid are prefix named calls (PRD §7.1: no infix operators,
no precedence). The built-ins below are the closed, reserved set. Each has a
fixed monomorphic signature except the list intrinsics, which are *element-type
resolved* (the only built-in polymorphism — distinct from user-facing ad-hoc
overloading, which v1 forbids). Type variables `#a`/`#b` are resolved by the
type checker's one-shot matcher.

Totality (PRD §7.5): every built-in is total. Integer `div`/`mod` by zero are
*defined* to be 0 rather than faulting, so well-typed programs never raise at
runtime for arithmetic. Partial list access is exposed only through total
`*_or` variants that take a default.

Higher-order built-ins (`map`, `filter`, `foldl`) receive an evaluation context
`ctx` exposing `apply(fnval, args)` and `charge(n)` for the resource budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .types import INT, BOOL, STR, TList, TFn, TVar, Type
from .values import FnVal

_A = TVar("a")
_B = TVar("b")
_C = TVar("c")


@dataclass(frozen=True)
class Builtin:
    name: str
    params: tuple[Type, ...]
    ret: Type
    arity: int
    fn: Callable[[list, Any], Any]


BUILTINS: dict[str, Builtin] = {}


def _reg(name: str, params: tuple[Type, ...], ret: Type, fn: Callable[[list, Any], Any]) -> None:
    BUILTINS[name] = Builtin(name, params, ret, len(params), fn)


# ---- Integer arithmetic (total) ------------------------------------------
_reg("add", (INT, INT), INT, lambda a, c: a[0] + a[1])
_reg("sub", (INT, INT), INT, lambda a, c: a[0] - a[1])
_reg("mul", (INT, INT), INT, lambda a, c: a[0] * a[1])
_reg("div", (INT, INT), INT, lambda a, c: 0 if a[1] == 0 else a[0] // a[1])
_reg("mod", (INT, INT), INT, lambda a, c: 0 if a[1] == 0 else a[0] % a[1])
_reg("neg", (INT,), INT, lambda a, c: -a[0])
_reg("abs", (INT,), INT, lambda a, c: abs(a[0]))
_reg("min", (INT, INT), INT, lambda a, c: min(a[0], a[1]))
_reg("max", (INT, INT), INT, lambda a, c: max(a[0], a[1]))

# ---- Integer comparisons -------------------------------------------------
_reg("lt_int", (INT, INT), BOOL, lambda a, c: a[0] < a[1])
_reg("le_int", (INT, INT), BOOL, lambda a, c: a[0] <= a[1])
_reg("gt_int", (INT, INT), BOOL, lambda a, c: a[0] > a[1])
_reg("ge_int", (INT, INT), BOOL, lambda a, c: a[0] >= a[1])
_reg("eq_int", (INT, INT), BOOL, lambda a, c: a[0] == a[1])
_reg("ne_int", (INT, INT), BOOL, lambda a, c: a[0] != a[1])

# ---- Booleans ------------------------------------------------------------
_reg("and", (BOOL, BOOL), BOOL, lambda a, c: a[0] and a[1])
_reg("or", (BOOL, BOOL), BOOL, lambda a, c: a[0] or a[1])
_reg("not", (BOOL,), BOOL, lambda a, c: not a[0])
_reg("xor", (BOOL, BOOL), BOOL, lambda a, c: a[0] != a[1])
_reg("eq_bool", (BOOL, BOOL), BOOL, lambda a, c: a[0] == a[1])
_reg("ne_bool", (BOOL, BOOL), BOOL, lambda a, c: a[0] != a[1])

# ---- Strings -------------------------------------------------------------
_reg("concat", (STR, STR), STR, lambda a, c: a[0] + a[1])
_reg("len_str", (STR,), INT, lambda a, c: len(a[0]))
_reg("eq_str", (STR, STR), BOOL, lambda a, c: a[0] == a[1])
_reg("ne_str", (STR, STR), BOOL, lambda a, c: a[0] != a[1])
_reg("to_str_int", (INT,), STR, lambda a, c: str(a[0]))
_reg("to_str_bool", (BOOL,), STR, lambda a, c: "true" if a[0] else "false")


# ---- List intrinsics (element-type resolved) -----------------------------
def _length(a, c):
    return len(a[0])


def _is_empty(a, c):
    return len(a[0]) == 0


def _append(a, c):
    c.charge(len(a[0]) + 1)
    return a[0] + [a[1]]


def _concat_list(a, c):
    c.charge(len(a[0]) + len(a[1]))
    return a[0] + a[1]


def _reverse(a, c):
    c.charge(len(a[0]))
    return list(reversed(a[0]))


def _get_or(a, c):
    lst, idx, default = a[0], a[1], a[2]
    return lst[idx] if 0 <= idx < len(lst) else default


def _head_or(a, c):
    lst, default = a[0], a[1]
    return lst[0] if lst else default


def _range(a, c):
    lo, hi = a[0], a[1]
    count = hi - lo
    if count <= 0:
        return []
    c.charge(count)  # resource budget guards against huge ranges
    return list(range(lo, hi))


def _map(a, c):
    lst, fn = a[0], a[1]
    c.charge(len(lst))
    return [c.apply(fn, [x]) for x in lst]


def _filter(a, c):
    lst, fn = a[0], a[1]
    c.charge(len(lst))
    return [x for x in lst if c.apply(fn, [x])]


def _foldl(a, c):
    lst, init, fn = a[0], a[1], a[2]
    c.charge(len(lst))
    acc = init
    for x in lst:
        acc = c.apply(fn, [acc, x])
    return acc


_reg("length", (TList(_A),), INT, _length)
_reg("is_empty", (TList(_A),), BOOL, _is_empty)
_reg("append", (TList(_A), _A), TList(_A), _append)
_reg("concat_list", (TList(_A), TList(_A)), TList(_A), _concat_list)
_reg("reverse", (TList(_A),), TList(_A), _reverse)
_reg("get_or", (TList(_A), INT, _A), _A, _get_or)
_reg("head_or", (TList(_A), _A), _A, _head_or)
_reg("range", (INT, INT), TList(INT), _range)
_reg("map", (TList(_A), TFn((_A,), _B)), TList(_B), _map)
_reg("filter", (TList(_A), TFn((_A,), BOOL)), TList(_A), _filter)
_reg("foldl", (TList(_A), _B, TFn((_B, _A), _B)), _B, _foldl)


# ---- Direct application of first-class function values --------------------
_reg("apply1", (TFn((_A,), _B), _A), _B, lambda a, c: c.apply(a[0], [a[1]]))
_reg(
    "apply2",
    (TFn((_A, _B), _C), _A, _B),
    _C,
    lambda a, c: c.apply(a[0], [a[1], a[2]]),
)


def is_builtin(name: str) -> bool:
    return name in BUILTINS
