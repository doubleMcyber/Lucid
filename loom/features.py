"""Feature extraction, grammar-production coverage, and difficulty scoring.

One AST walk produces:
  * `features(program)`   — human-readable tags for the dataset record and spec.
  * `productions(program)`— the set of grammar productions exercised (for the
                            coverage metric, PRD §4 "grammar-production coverage").
  * `difficulty(program)` — an integer used by the curriculum schedule.
  * `type_interactions`   — pairs like ("foreach","#List[#Int]") for the
                            type-interaction coverage report (PRD §4).

`ALL_PRODUCTIONS` is the universe coverage is measured against.
"""

from __future__ import annotations

from lucid.ast import (
    BoolLit, BuiltinCall, Call, Cond, Expr, Field, FnDecl, Foreach, IfStmt,
    IntLit, Let, ListLit, Match, Program, RecordDecl, RecordLit, Return,
    SetStmt, Stmt, StrLit, VarDecl, VarRef, FnRef, VariantDecl, VariantLit,
)
from lucid.builtins_def import BUILTINS
from lucid.types import TFn, TList, TRecord, TVariant, Type


_EXPR_PRODUCTIONS = [
    "expr:int", "expr:bool", "expr:str", "expr:var", "expr:fnref",
    "expr:builtin", "expr:call", "expr:list", "expr:record", "expr:field",
    "expr:variant", "expr:match", "expr:cond",
]
_STMT_PRODUCTIONS = [
    "stmt:let", "stmt:var", "stmt:set", "stmt:foreach", "stmt:if", "stmt:return",
]
_DECL_PRODUCTIONS = ["decl:record", "decl:variant", "decl:fn"]
_TYPE_PRODUCTIONS = [
    "type:int", "type:bool", "type:str", "type:list", "type:fn",
    "type:record", "type:variant",
]
_BUILTIN_PRODUCTIONS = [f"builtin:{n}" for n in sorted(BUILTINS)]

ALL_PRODUCTIONS: frozenset[str] = frozenset(
    _EXPR_PRODUCTIONS + _STMT_PRODUCTIONS + _DECL_PRODUCTIONS
    + _TYPE_PRODUCTIONS + _BUILTIN_PRODUCTIONS
)


def _type_productions(t: Type, out: set[str]) -> None:
    from lucid.types import TInt, TBool, TStr
    if isinstance(t, TInt):
        out.add("type:int")
    elif isinstance(t, TBool):
        out.add("type:bool")
    elif isinstance(t, TStr):
        out.add("type:str")
    elif isinstance(t, TList):
        out.add("type:list")
        _type_productions(t.elem, out)
    elif isinstance(t, TFn):
        out.add("type:fn")
        for p in t.params:
            _type_productions(p, out)
        _type_productions(t.ret, out)
    elif isinstance(t, TRecord):
        out.add("type:record")
    elif isinstance(t, TVariant):
        out.add("type:variant")


class _Walk:
    def __init__(self):
        self.prods: set[str] = set()
        self.features: set[str] = set()
        self.interactions: set[tuple[str, str]] = set()
        self.n_nodes = 0
        self.max_depth = 0

    def expr(self, e: Expr, depth: int) -> None:
        self.n_nodes += 1
        self.max_depth = max(self.max_depth, depth)
        if isinstance(e, IntLit):
            self.prods.add("expr:int")
        elif isinstance(e, BoolLit):
            self.prods.add("expr:bool")
        elif isinstance(e, StrLit):
            self.prods.add("expr:str"); self.features.add("strings")
        elif isinstance(e, VarRef):
            self.prods.add("expr:var")
        elif isinstance(e, FnRef):
            self.prods.add("expr:fnref"); self.features.add("first_class_fn")
        elif isinstance(e, BuiltinCall):
            self.prods.add("expr:builtin")
            self.prods.add(f"builtin:{e.name}")
            self.features.add(f"builtin_{e.name}")
            if e.name in ("map", "filter", "foldl", "apply1", "apply2"):
                self.features.add("higher_order")
            for a in e.args:
                self.expr(a, depth + 1)
        elif isinstance(e, Call):
            self.prods.add("expr:call"); self.features.add("user_call")
            for a in e.args:
                self.expr(a, depth + 1)
        elif isinstance(e, ListLit):
            self.prods.add("expr:list")
            _type_productions(TList(e.elem_ty), self.prods)
            for a in e.elems:
                self.expr(a, depth + 1)
        elif isinstance(e, RecordLit):
            self.prods.add("expr:record"); self.features.add("record")
            for _, v in e.fields:
                self.expr(v, depth + 1)
        elif isinstance(e, Field):
            self.prods.add("expr:field")
            self.expr(e.record, depth + 1)
        elif isinstance(e, VariantLit):
            self.prods.add("expr:variant"); self.features.add("variant")
            for a in e.args:
                self.expr(a, depth + 1)
        elif isinstance(e, Match):
            self.prods.add("expr:match"); self.features.add("match")
            self.expr(e.scrutinee, depth + 1)
            for a in e.arms:
                self.expr(a.body, depth + 1)
        elif isinstance(e, Cond):
            self.prods.add("expr:cond"); self.features.add("cond")
            self.expr(e.test, depth + 1)
            self.expr(e.then, depth + 1)
            self.expr(e.els, depth + 1)

    def stmt(self, s: Stmt, depth: int) -> None:
        self.n_nodes += 1
        if isinstance(s, Let):
            self.prods.add("stmt:let")
            _type_productions(s.ty, self.prods)
            self.expr(s.expr, depth + 1)
        elif isinstance(s, VarDecl):
            self.prods.add("stmt:var"); self.features.add("mutable_local")
            _type_productions(s.ty, self.prods)
            self.expr(s.expr, depth + 1)
        elif isinstance(s, SetStmt):
            self.prods.add("stmt:set")
            self.expr(s.expr, depth + 1)
        elif isinstance(s, Foreach):
            self.prods.add("stmt:foreach"); self.features.add("foreach")
            _type_productions(s.elem_ty, self.prods)
            self.interactions.add(("foreach", str(TList(s.elem_ty))))
            self.expr(s.iter, depth + 1)
            for b in s.body:
                self.stmt(b, depth + 1)
        elif isinstance(s, IfStmt):
            self.prods.add("stmt:if"); self.features.add("if")
            self.expr(s.cond, depth + 1)
            for b in s.then_body:
                self.stmt(b, depth + 1)
            for b in s.else_body:
                self.stmt(b, depth + 1)
        elif isinstance(s, Return):
            self.prods.add("stmt:return")
            self.expr(s.expr, depth + 1)

    def decl(self, d) -> None:
        if isinstance(d, RecordDecl):
            self.prods.add("decl:record"); self.features.add("record")
            for _, t in d.fields:
                _type_productions(t, self.prods)
        elif isinstance(d, VariantDecl):
            self.prods.add("decl:variant"); self.features.add("variant")
            for _, tys in d.ctors:
                for t in tys:
                    _type_productions(t, self.prods)
        elif isinstance(d, FnDecl):
            self.prods.add("decl:fn")
            for _, t in d.params:
                _type_productions(t, self.prods)
            _type_productions(d.ret, self.prods)
            for s in d.body:
                self.stmt(s, 1)


def analyze(program: Program) -> _Walk:
    w = _Walk()
    for d in program.decls:
        w.decl(d)
    return w


def features(program: Program) -> set[str]:
    return analyze(program).features


def productions(program: Program) -> set[str]:
    return analyze(program).prods


def difficulty(program: Program) -> int:
    w = analyze(program)
    # Three bounded contributions (size, nesting, advanced features) so the full
    # 1-10 range is exercised and difficulty correlates with curriculum stage,
    # instead of saturating at 10 for every feature-rich program.
    advanced = w.features & {
        "higher_order", "match", "variant", "record", "cond", "first_class_fn",
        "mutable_local", "foreach", "if",
    }
    size_term = min(w.n_nodes // 30, 4)
    depth_term = min(w.max_depth // 4, 2)
    feature_term = min(len(advanced), 3)
    score = 1 + size_term + depth_term + feature_term
    return max(1, min(score, 10))
