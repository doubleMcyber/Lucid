"""Canonical printer for Lucid (PRD §7.2).

Produces the *unique* surface text for an AST. Combined with the parser this
gives the bijection the whole project rests on:

    parse(print(ast)) == ast                 (round-trip through text)
    print(parse(canonical_text)) == text     (canonical text is a fixed point)

Conventions that define "canonical":
  * 2-space indentation; statements are one-per-line; declarations separated by
    a blank line.
  * Expressions render on a single line (whitespace is non-semantic, so nesting
    needs no layout bookkeeping).
  * Strings use the minimal hash-fence so there is exactly one spelling.
  * Record-literal fields are emitted sorted by field name (an order intrinsic
    to the node, requiring no TypeEnv), so two semantically equal records are
    textually identical and share an ast_hash.

The printer is a pure function of the AST and needs no TypeEnv: every node
already carries the type annotations and field order required to print it.
"""

from __future__ import annotations

from .ast import (
    BoolLit, BuiltinCall, Call, Cond, Decl, Expr, Field, FnDecl, FnRef,
    Foreach, IfStmt, IntLit, Let, ListLit, Match, Program, RecordDecl,
    RecordLit, Return, SetStmt, Stmt, StrLit, VarDecl, VarRef, VariantDecl,
    VariantLit,
)
from .types import Type

INDENT = "  "


def fence_string(s: str) -> str:
    """Minimal-hash raw fence for a string literal (escape-free, unique)."""
    k = 0
    while ('"' + "#" * k) in s:
        k += 1
    hashes = "#" * k
    return f'{hashes}"{s}"{hashes}'


def print_type(t: Type) -> str:
    return str(t)


def print_expr(e: Expr) -> str:
    if isinstance(e, IntLit):
        if e.value < 0:
            raise ValueError("integer literals must be non-negative; use neg(...)")
        return str(e.value)
    if isinstance(e, BoolLit):
        return "true" if e.value else "false"
    if isinstance(e, StrLit):
        return fence_string(e.value)
    if isinstance(e, VarRef):
        return f"${e.name}"
    if isinstance(e, FnRef):
        return f"@{e.name}"
    if isinstance(e, BuiltinCall):
        return f"{e.name}({_args(e.args)})"
    if isinstance(e, Call):
        return f"@{e.name}({_args(e.args)})"
    if isinstance(e, ListLit):
        return f"list[{print_type(e.elem_ty)}]({_args(e.elems)})"
    if isinstance(e, RecordLit):
        if not e.fields:
            return f"new #{e.type_name} {{}}"
        # Canonical record-literal field order is sorted by field name. This is
        # intrinsic to the node (needs no TypeEnv), so the printer is the single
        # canonicalizing authority: two literals that differ only in field order
        # print identically and therefore share an ast_hash.
        ordered = sorted(e.fields, key=lambda nv: nv[0])
        inner = ", ".join(f"%{n} = {print_expr(v)}" for n, v in ordered)
        return f"new #{e.type_name} {{ {inner} }}"
    if isinstance(e, Field):
        return f"get %{e.field}({print_expr(e.record)})"
    if isinstance(e, VariantLit):
        return f"tag #{e.type_name} {e.tag}({_args(e.args)})"
    if isinstance(e, Match):
        arms = " ".join(
            f"case {a.tag}({_binders(a.binders)}) -> {print_expr(a.body)}"
            for a in e.arms
        )
        return f"match #{e.type_name} {print_expr(e.scrutinee)} of {arms} end match"
    if isinstance(e, Cond):
        return (
            f"cond {print_expr(e.test)} then {print_expr(e.then)} "
            f"else {print_expr(e.els)} end cond"
        )
    raise TypeError(f"cannot print expr node {type(e).__name__}")


def _args(args: tuple[Expr, ...]) -> str:
    return ", ".join(print_expr(a) for a in args)


def _binders(binders: tuple[tuple[str, Type], ...]) -> str:
    return ", ".join(f"${n} : {print_type(t)}" for n, t in binders)


def print_stmt(s: Stmt, level: int) -> list[str]:
    pad = INDENT * level
    if isinstance(s, Let):
        return [f"{pad}let ${s.name} : {print_type(s.ty)} = {print_expr(s.expr)} ;"]
    if isinstance(s, VarDecl):
        return [f"{pad}var ${s.name} : {print_type(s.ty)} = {print_expr(s.expr)} ;"]
    if isinstance(s, SetStmt):
        return [f"{pad}set ${s.name} = {print_expr(s.expr)} ;"]
    if isinstance(s, Foreach):
        head = (
            f"{pad}foreach ${s.var} : {print_type(s.elem_ty)} "
            f"in {print_expr(s.iter)} do"
        )
        lines = [head]
        for st in s.body:
            lines.extend(print_stmt(st, level + 1))
        lines.append(f"{pad}end foreach ;")
        return lines
    if isinstance(s, IfStmt):
        lines = [f"{pad}if {print_expr(s.cond)} do"]
        for st in s.then_body:
            lines.extend(print_stmt(st, level + 1))
        lines.append(f"{pad}else")
        for st in s.else_body:
            lines.extend(print_stmt(st, level + 1))
        lines.append(f"{pad}end if ;")
        return lines
    if isinstance(s, Return):
        return [f"{pad}return {print_expr(s.expr)} ;"]
    raise TypeError(f"cannot print stmt node {type(s).__name__}")


def print_decl(d: Decl) -> str:
    if isinstance(d, RecordDecl):
        fields = ", ".join(f"%{n} : {print_type(t)}" for n, t in d.fields)
        if not d.fields:
            return f"record #{d.name} = {{}} ;"
        return f"record #{d.name} = {{ {fields} }} ;"
    if isinstance(d, VariantDecl):
        ctors = " | ".join(
            f"{tag}({', '.join(print_type(t) for t in tys)})" for tag, tys in d.ctors
        )
        return f"variant #{d.name} = {ctors} ;"
    if isinstance(d, FnDecl):
        params = ", ".join(f"${n} : {print_type(t)}" for n, t in d.params)
        head = f"fn @{d.name} ({params}) -> {print_type(d.ret)} = do"
        lines = [head]
        for st in d.body:
            lines.extend(print_stmt(st, 1))
        lines.append(f"end @{d.name} ;")
        return "\n".join(lines)
    raise TypeError(f"cannot print decl node {type(d).__name__}")


def print_program(p: Program) -> str:
    return "\n\n".join(print_decl(d) for d in p.decls)
