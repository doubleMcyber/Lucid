"""Lucid abstract syntax tree.

Every node is a frozen dataclass so ASTs are value-comparable (the bijectivity
property test relies on `parse(print(ast)) == ast`) and hashable where useful.
Lists inside nodes are stored as tuples to preserve immutability/hashability.

The AST is the single source of truth. The canonical printer is a pure function
of the AST; the parser is its inverse on canonical text.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Type


# --------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------
class Expr:
    pass


@dataclass(frozen=True)
class IntLit(Expr):
    value: int


@dataclass(frozen=True)
class BoolLit(Expr):
    value: bool


@dataclass(frozen=True)
class StrLit(Expr):
    value: str


@dataclass(frozen=True)
class VarRef(Expr):
    """A local reference: `$name`."""

    name: str


@dataclass(frozen=True)
class FnRef(Expr):
    """A first-class function value: `@name` used in expression position."""

    name: str


@dataclass(frozen=True)
class BuiltinCall(Expr):
    """A reserved-name prefix call: `add($a, $b)`."""

    name: str
    args: tuple[Expr, ...]


@dataclass(frozen=True)
class Call(Expr):
    """A user-function prefix call: `@f($a, $b)`."""

    name: str
    args: tuple[Expr, ...]


@dataclass(frozen=True)
class ListLit(Expr):
    """`list[#T]($e1, $e2, ...)` — homogeneous list with explicit element type."""

    elem_ty: Type
    elems: tuple[Expr, ...]


@dataclass(frozen=True)
class RecordLit(Expr):
    """`new #Rec { %f = e, ... }` — fields in declaration order (canonical)."""

    type_name: str
    fields: tuple[tuple[str, Expr], ...]


@dataclass(frozen=True)
class Field(Expr):
    """`get %f ( e )` — project a field out of a record."""

    field: str
    record: Expr


@dataclass(frozen=True)
class VariantLit(Expr):
    """`tag #Var Tag ( args )` — construct a tagged-variant value."""

    type_name: str
    tag: str
    args: tuple[Expr, ...]


@dataclass(frozen=True)
class MatchArm:
    tag: str
    # binders: one (name, type) per payload slot of the tag, in order.
    binders: tuple[tuple[str, Type], ...]
    body: Expr


@dataclass(frozen=True)
class Match(Expr):
    """`match e of case Tag(...) -> e ... end match` — exhaustive variant match."""

    type_name: str
    scrutinee: Expr
    arms: tuple[MatchArm, ...]


@dataclass(frozen=True)
class Cond(Expr):
    """`cond <test> then <e> else <e> end cond` — lazy expression-level if."""

    test: Expr
    then: Expr
    els: Expr


# --------------------------------------------------------------------------
# Statements
# --------------------------------------------------------------------------
class Stmt:
    pass


@dataclass(frozen=True)
class Let(Stmt):
    name: str
    ty: Type
    expr: Expr


@dataclass(frozen=True)
class VarDecl(Stmt):
    name: str
    ty: Type
    expr: Expr


@dataclass(frozen=True)
class SetStmt(Stmt):
    name: str
    expr: Expr


@dataclass(frozen=True)
class Foreach(Stmt):
    var: str
    elem_ty: Type
    iter: Expr
    body: tuple[Stmt, ...]


@dataclass(frozen=True)
class IfStmt(Stmt):
    cond: Expr
    then_body: tuple[Stmt, ...]
    else_body: tuple[Stmt, ...]


@dataclass(frozen=True)
class Return(Stmt):
    expr: Expr


# --------------------------------------------------------------------------
# Declarations and program
# --------------------------------------------------------------------------
class Decl:
    pass


@dataclass(frozen=True)
class RecordDecl(Decl):
    name: str
    fields: tuple[tuple[str, Type], ...]


@dataclass(frozen=True)
class VariantDecl(Decl):
    name: str
    ctors: tuple[tuple[str, tuple[Type, ...]], ...]


@dataclass(frozen=True)
class FnDecl(Decl):
    name: str
    params: tuple[tuple[str, Type], ...]
    ret: Type
    body: tuple[Stmt, ...]


@dataclass(frozen=True)
class Program(Decl):
    decls: tuple[Decl, ...]

    def functions(self) -> list[FnDecl]:
        return [d for d in self.decls if isinstance(d, FnDecl)]

    def entry(self) -> FnDecl:
        """The entry point is the last declared function."""
        fns = self.functions()
        if not fns:
            raise ValueError("program has no functions")
        return fns[-1]
