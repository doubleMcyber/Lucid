"""Immutable AST traversal/rewrite helpers used by the repair and refactor
labelers. Lucid ASTs are frozen, so every rewrite returns a new tree.

The core is a deterministic pre-order walk over *every* expression node in a
program (including those nested in statements), with the ability to replace the
k-th node matching a predicate. Repair mutations and refactor transforms are
both built on top of this.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from lucid.ast import (
    BuiltinCall, Call, Cond, Expr, Field, FnDecl, Foreach, IfStmt, ListLit,
    Let, Match, MatchArm, Program, RecordLit, Return, SetStmt, Stmt, VarDecl,
    VarRef, VariantLit,
)


# --------------------------------------------------------------------------
# Mapping immediate sub-expressions
# --------------------------------------------------------------------------
def map_sub_exprs(e: Expr, f: Callable[[Expr], Expr]) -> Expr:
    if isinstance(e, BuiltinCall):
        return replace(e, args=tuple(f(a) for a in e.args))
    if isinstance(e, Call):
        return replace(e, args=tuple(f(a) for a in e.args))
    if isinstance(e, ListLit):
        return replace(e, elems=tuple(f(a) for a in e.elems))
    if isinstance(e, RecordLit):
        return replace(e, fields=tuple((n, f(v)) for n, v in e.fields))
    if isinstance(e, Field):
        return replace(e, record=f(e.record))
    if isinstance(e, VariantLit):
        return replace(e, args=tuple(f(a) for a in e.args))
    if isinstance(e, Match):
        return replace(
            e,
            scrutinee=f(e.scrutinee),
            arms=tuple(replace(a, body=f(a.body)) for a in e.arms),
        )
    if isinstance(e, Cond):
        return replace(e, test=f(e.test), then=f(e.then), els=f(e.els))
    # leaves: IntLit, BoolLit, StrLit, VarRef, FnRef
    return e


# --------------------------------------------------------------------------
# Pre-order transform over all exprs
# --------------------------------------------------------------------------
def _tx_expr(e: Expr, f: Callable[[Expr], Expr]) -> Expr:
    e2 = f(e)
    return map_sub_exprs(e2, lambda c: _tx_expr(c, f))


def _tx_stmt(s: Stmt, f: Callable[[Expr], Expr]) -> Stmt:
    if isinstance(s, (Let, VarDecl)):
        return replace(s, expr=_tx_expr(s.expr, f))
    if isinstance(s, SetStmt):
        return replace(s, expr=_tx_expr(s.expr, f))
    if isinstance(s, Foreach):
        return replace(s, iter=_tx_expr(s.iter, f),
                       body=tuple(_tx_stmt(b, f) for b in s.body))
    if isinstance(s, IfStmt):
        return replace(s, cond=_tx_expr(s.cond, f),
                       then_body=tuple(_tx_stmt(b, f) for b in s.then_body),
                       else_body=tuple(_tx_stmt(b, f) for b in s.else_body))
    if isinstance(s, Return):
        return replace(s, expr=_tx_expr(s.expr, f))
    return s


def transform_program_exprs(program: Program, f: Callable[[Expr], Expr]) -> Program:
    new_decls = []
    for d in program.decls:
        if isinstance(d, FnDecl):
            new_decls.append(replace(d, body=tuple(_tx_stmt(s, f) for s in d.body)))
        else:
            new_decls.append(d)
    return Program(tuple(new_decls))


def count_exprs(program: Program, pred: Callable[[Expr], bool]) -> int:
    n = 0

    def f(e: Expr) -> Expr:
        nonlocal n
        if pred(e):
            n += 1
        return e

    transform_program_exprs(program, f)
    return n


def replace_nth_expr(program: Program, pred: Callable[[Expr], bool], index: int,
                     newfn: Callable[[Expr], Expr]) -> Program:
    counter = 0

    def f(e: Expr) -> Expr:
        nonlocal counter
        if pred(e):
            i = counter
            counter += 1
            if i == index:
                return newfn(e)
        return e

    return transform_program_exprs(program, f)


# --------------------------------------------------------------------------
# Local-variable renaming (semantics-preserving refactor primitive)
# --------------------------------------------------------------------------
def rename_local_in_fn(fn: FnDecl, old: str, new: str) -> FnDecl:
    """Rename a local/binder `old` to `new` throughout one function body.
    Names are function-local and unique, so a straight substitution is safe."""

    def fe(e: Expr) -> Expr:
        if isinstance(e, VarRef) and e.name == old:
            return VarRef(new)
        if isinstance(e, Match):
            # rename match-arm binder *declarations* too, not just their uses
            return replace(
                e,
                scrutinee=fe(e.scrutinee),
                arms=tuple(
                    replace(
                        a,
                        binders=tuple((new if bn == old else bn, bt) for bn, bt in a.binders),
                        body=fe(a.body),
                    )
                    for a in e.arms
                ),
            )
        return map_sub_exprs(e, fe)

    def fs(s: Stmt) -> Stmt:
        if isinstance(s, (Let, VarDecl)):
            nm = new if s.name == old else s.name
            return replace(s, name=nm, expr=fe(s.expr))
        if isinstance(s, SetStmt):
            nm = new if s.name == old else s.name
            return replace(s, name=nm, expr=fe(s.expr))
        if isinstance(s, Foreach):
            var = new if s.var == old else s.var
            return replace(s, var=var, iter=fe(s.iter),
                           body=tuple(fs(b) for b in s.body))
        if isinstance(s, IfStmt):
            return replace(s, cond=fe(s.cond),
                           then_body=tuple(fs(b) for b in s.then_body),
                           else_body=tuple(fs(b) for b in s.else_body))
        if isinstance(s, Return):
            return replace(s, expr=fe(s.expr))
        return s

    params = tuple((new if n == old else n, t) for n, t in fn.params)
    return replace(fn, params=params, body=tuple(fs(s) for s in fn.body))


def collect_local_names(fn: FnDecl) -> list[str]:
    names: list[str] = [n for n, _ in fn.params]

    def fe(e: Expr) -> Expr:
        if isinstance(e, Match):
            for a in e.arms:
                for bn, _ in a.binders:
                    names.append(bn)
        map_sub_exprs(e, fe)
        return e

    def fs(s: Stmt) -> None:
        if isinstance(s, (Let, VarDecl)):
            names.append(s.name)
            fe(s.expr)
        elif isinstance(s, SetStmt):
            fe(s.expr)
        elif isinstance(s, Foreach):
            names.append(s.var)
            fe(s.iter)
            for b in s.body:
                fs(b)
        elif isinstance(s, IfStmt):
            fe(s.cond)
            for b in s.then_body:
                fs(b)
            for b in s.else_body:
                fs(b)
        elif isinstance(s, Return):
            fe(s.expr)

    for s in fn.body:
        fs(s)
    # dedup, preserve order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out
