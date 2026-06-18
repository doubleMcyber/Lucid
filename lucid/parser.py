"""Lucid recursive-descent parser.

The grammar has no optional syntax and no infix operators, so the parser is a
straightforward LL(1) descent with zero backtracking. Whitespace is already
gone (handled by the lexer), so structure is fully explicit: every block opens
with `do`/`{`/`[` and closes with a self-naming terminator (`end @f`,
`end foreach`, `end match`, `end cond`).

User type references (`#Name`) are resolved to `TRecord`/`TVariant` against a
`TypeEnv` built incrementally from the declarations — so types must be declared
before use (the sampler and the canonical printer both honor this). Record
literals are normalized to declaration field order here, which is what makes
"single canonical form per program" hold even for non-canonical input.

`parse()` returns a `Module` bundling the `Program` AST with its `TypeEnv`.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import lexer as lx
from .ast import (
    BoolLit, BuiltinCall, Call, Cond, Decl, Expr, Field, FnDecl, FnRef,
    Foreach, IfStmt, IntLit, Let, ListLit, Match, MatchArm, Program,
    RecordDecl, RecordLit, Return, SetStmt, Stmt, StrLit, VarDecl, VarRef,
    VariantDecl, VariantLit,
)
from .errors import ParseError
from .keywords import is_keyword
from .types import (
    BOOL, INT, STR, RecordDef, TFn, TList, TRecord, TVariant, Type,
    TypeEnv, VariantDef,
)


@dataclass
class Module:
    program: Program
    tenv: TypeEnv


class Parser:
    def __init__(self, toks: list[lx.Token]):
        self.toks = toks
        self.i = 0
        self.tenv = TypeEnv()

    # -- token helpers -----------------------------------------------------
    def peek(self, off: int = 0) -> lx.Token:
        j = self.i + off
        return self.toks[j] if j < len(self.toks) else self.toks[-1]

    def at(self, kind: str) -> bool:
        return self.peek().kind == kind

    def at_kw(self, kw: str) -> bool:
        t = self.peek()
        return t.kind == lx.NAME and t.value == kw

    def advance(self) -> lx.Token:
        t = self.peek()
        self.i += 1
        return t

    def eat(self, kind: str) -> lx.Token:
        t = self.peek()
        if t.kind != kind:
            raise ParseError(f"expected {kind}, got {t.kind} {t.value!r}", self.i)
        return self.advance()

    def eat_kw(self, kw: str) -> lx.Token:
        t = self.peek()
        if not (t.kind == lx.NAME and t.value == kw):
            raise ParseError(f"expected keyword '{kw}', got {t.value!r}", self.i)
        return self.advance()

    # -- entry -------------------------------------------------------------
    def parse_module(self) -> Module:
        decls: list[Decl] = []
        while not self.at(lx.EOF):
            decls.append(self.parse_decl())
        return Module(Program(tuple(decls)), self.tenv)

    def parse_decl(self) -> Decl:
        if self.at_kw("record"):
            return self.parse_record_decl()
        if self.at_kw("variant"):
            return self.parse_variant_decl()
        if self.at_kw("fn"):
            return self.parse_fn_decl()
        t = self.peek()
        raise ParseError(f"expected declaration, got {t.value!r}", self.i)

    # -- declarations ------------------------------------------------------
    def parse_record_decl(self) -> RecordDecl:
        self.eat_kw("record")
        name = self.eat(lx.TYPE).value
        self.eat(lx.EQ)
        self.eat(lx.LBRACE)
        fields: list[tuple[str, Type]] = []
        if not self.at(lx.RBRACE):
            while True:
                fname = self.eat(lx.FIELD).value
                self.eat(lx.COLON)
                fty = self.parse_type()
                fields.append((fname, fty))
                if self.at(lx.COMMA):
                    self.advance()
                    continue
                break
        self.eat(lx.RBRACE)
        self.eat(lx.SEMI)
        self.tenv.add_record(RecordDef(name, fields))
        return RecordDecl(name, tuple(fields))

    def parse_variant_decl(self) -> VariantDecl:
        self.eat_kw("variant")
        name = self.eat(lx.TYPE).value
        self.eat(lx.EQ)
        ctors: list[tuple[str, tuple[Type, ...]]] = []
        ctors.append(self.parse_ctor())
        while self.at(lx.PIPE):
            self.advance()
            ctors.append(self.parse_ctor())
        self.eat(lx.SEMI)
        self.tenv.add_variant(
            VariantDef(name, [(t, list(ts)) for t, ts in ctors])
        )
        return VariantDecl(name, tuple(ctors))

    def parse_ctor(self) -> tuple[str, tuple[Type, ...]]:
        tag = self.eat(lx.NAME).value
        if is_keyword(tag):
            raise ParseError(f"variant tag may not be a keyword: {tag!r}", self.i)
        self.eat(lx.LPAREN)
        tys: list[Type] = []
        if not self.at(lx.RPAREN):
            while True:
                tys.append(self.parse_type())
                if self.at(lx.COMMA):
                    self.advance()
                    continue
                break
        self.eat(lx.RPAREN)
        return (tag, tuple(tys))

    def parse_fn_decl(self) -> FnDecl:
        self.eat_kw("fn")
        name = self.eat(lx.FUNC).value
        self.eat(lx.LPAREN)
        params: list[tuple[str, Type]] = []
        if not self.at(lx.RPAREN):
            while True:
                pname = self.eat(lx.LOCAL).value
                self.eat(lx.COLON)
                pty = self.parse_type()
                params.append((pname, pty))
                if self.at(lx.COMMA):
                    self.advance()
                    continue
                break
        self.eat(lx.RPAREN)
        self.eat(lx.ARROW)
        ret = self.parse_type()
        self.eat(lx.EQ)
        self.eat_kw("do")
        body = self.parse_stmts(("end",))
        self.eat_kw("end")
        close = self.eat(lx.FUNC).value
        if close != name:
            raise ParseError(
                f"block terminator 'end @{close}' does not match 'fn @{name}'",
                self.i,
            )
        self.eat(lx.SEMI)
        return FnDecl(name, tuple(params), ret, tuple(body))

    # -- types -------------------------------------------------------------
    def parse_type(self) -> Type:
        name = self.eat(lx.TYPE).value
        if name == "Int":
            return INT
        if name == "Bool":
            return BOOL
        if name == "Str":
            return STR
        if name == "List":
            self.eat(lx.LBRACK)
            elem = self.parse_type()
            self.eat(lx.RBRACK)
            return TList(elem)
        if name == "Fn":
            self.eat(lx.LBRACK)
            self.eat(lx.LPAREN)
            ps: list[Type] = []
            if not self.at(lx.RPAREN):
                while True:
                    ps.append(self.parse_type())
                    if self.at(lx.COMMA):
                        self.advance()
                        continue
                    break
            self.eat(lx.RPAREN)
            self.eat(lx.ARROW)
            ret = self.parse_type()
            self.eat(lx.RBRACK)
            return TFn(tuple(ps), ret)
        # user-declared type
        if self.tenv.lookup_record(name) is not None:
            return TRecord(name)
        if self.tenv.lookup_variant(name) is not None:
            return TVariant(name)
        raise ParseError(f"unknown type #{name} (declare it before use)", self.i)

    # -- statements --------------------------------------------------------
    def parse_stmts(self, stop_kws: tuple[str, ...]) -> list[Stmt]:
        stmts: list[Stmt] = []
        while not (self.at(lx.NAME) and self.peek().value in stop_kws):
            if self.at(lx.EOF):
                raise ParseError("unexpected end of input inside block", self.i)
            stmts.append(self.parse_stmt())
        return stmts

    def parse_stmt(self) -> Stmt:
        if self.at_kw("let") or self.at_kw("var"):
            return self.parse_binding()
        if self.at_kw("set"):
            return self.parse_set()
        if self.at_kw("foreach"):
            return self.parse_foreach()
        if self.at_kw("if"):
            return self.parse_if()
        if self.at_kw("return"):
            return self.parse_return()
        t = self.peek()
        raise ParseError(f"expected statement, got {t.value!r}", self.i)

    def parse_binding(self) -> Stmt:
        kw = self.advance().value  # 'let' or 'var'
        name = self.eat(lx.LOCAL).value
        self.eat(lx.COLON)
        ty = self.parse_type()
        self.eat(lx.EQ)
        expr = self.parse_expr()
        self.eat(lx.SEMI)
        if kw == "let":
            return Let(name, ty, expr)
        return VarDecl(name, ty, expr)

    def parse_set(self) -> Stmt:
        self.eat_kw("set")
        name = self.eat(lx.LOCAL).value
        self.eat(lx.EQ)
        expr = self.parse_expr()
        self.eat(lx.SEMI)
        return SetStmt(name, expr)

    def parse_foreach(self) -> Stmt:
        self.eat_kw("foreach")
        var = self.eat(lx.LOCAL).value
        self.eat(lx.COLON)
        elem_ty = self.parse_type()
        self.eat_kw("in")
        it = self.parse_expr()
        self.eat_kw("do")
        body = self.parse_stmts(("end",))
        self.eat_kw("end")
        self.eat_kw("foreach")
        self.eat(lx.SEMI)
        return Foreach(var, elem_ty, it, tuple(body))

    def parse_if(self) -> Stmt:
        self.eat_kw("if")
        cond = self.parse_expr()
        self.eat_kw("do")
        then_body = self.parse_stmts(("else",))
        self.eat_kw("else")
        else_body = self.parse_stmts(("end",))
        self.eat_kw("end")
        self.eat_kw("if")
        self.eat(lx.SEMI)
        return IfStmt(cond, tuple(then_body), tuple(else_body))

    def parse_return(self) -> Stmt:
        self.eat_kw("return")
        expr = self.parse_expr()
        self.eat(lx.SEMI)
        return Return(expr)

    # -- expressions -------------------------------------------------------
    def parse_expr(self) -> Expr:
        t = self.peek()
        if t.kind == lx.INT:
            self.advance()
            return IntLit(int(t.value))
        if t.kind == lx.STR:
            self.advance()
            return StrLit(t.value)
        if t.kind == lx.LOCAL:
            self.advance()
            return VarRef(t.value)
        if t.kind == lx.FUNC:
            self.advance()
            if self.at(lx.LPAREN):
                args = self.parse_args()
                return Call(t.value, args)
            return FnRef(t.value)
        if t.kind == lx.NAME:
            return self.parse_name_expr()
        raise ParseError(f"expected expression, got {t.kind} {t.value!r}", self.i)

    def parse_name_expr(self) -> Expr:
        t = self.peek()
        kw = t.value
        if kw == "true":
            self.advance()
            return BoolLit(True)
        if kw == "false":
            self.advance()
            return BoolLit(False)
        if kw == "list":
            return self.parse_list_lit()
        if kw == "new":
            return self.parse_record_lit()
        if kw == "get":
            return self.parse_field()
        if kw == "tag":
            return self.parse_variant_lit()
        if kw == "match":
            return self.parse_match()
        if kw == "cond":
            return self.parse_cond()
        if is_keyword(kw):
            raise ParseError(f"unexpected keyword in expression: {kw!r}", self.i)
        # otherwise: a built-in call (bare name followed by parens)
        self.advance()
        if not self.at(lx.LPAREN):
            raise ParseError(
                f"bare name {kw!r} must be a built-in call with arguments", self.i
            )
        args = self.parse_args()
        return BuiltinCall(kw, args)

    def parse_args(self) -> tuple[Expr, ...]:
        self.eat(lx.LPAREN)
        args: list[Expr] = []
        if not self.at(lx.RPAREN):
            while True:
                args.append(self.parse_expr())
                if self.at(lx.COMMA):
                    self.advance()
                    continue
                break
        self.eat(lx.RPAREN)
        return tuple(args)

    def parse_list_lit(self) -> Expr:
        self.eat_kw("list")
        self.eat(lx.LBRACK)
        elem_ty = self.parse_type()
        self.eat(lx.RBRACK)
        elems = self.parse_args()
        return ListLit(elem_ty, elems)

    def parse_record_lit(self) -> Expr:
        self.eat_kw("new")
        tname = self.eat(lx.TYPE).value
        self.eat(lx.LBRACE)
        fields: list[tuple[str, Expr]] = []
        if not self.at(lx.RBRACE):
            while True:
                fname = self.eat(lx.FIELD).value
                self.eat(lx.EQ)
                fexpr = self.parse_expr()
                fields.append((fname, fexpr))
                if self.at(lx.COMMA):
                    self.advance()
                    continue
                break
        self.eat(lx.RBRACE)
        # Canonical record-literal field order is sorted by field name (intrinsic
        # to the node — matches the printer; no TypeEnv needed). Field-set
        # validity is left to the type checker.
        fields = sorted(fields, key=lambda nv: nv[0])
        return RecordLit(tname, tuple(fields))

    def parse_field(self) -> Expr:
        self.eat_kw("get")
        fname = self.eat(lx.FIELD).value
        self.eat(lx.LPAREN)
        rec = self.parse_expr()
        self.eat(lx.RPAREN)
        return Field(fname, rec)

    def parse_variant_lit(self) -> Expr:
        self.eat_kw("tag")
        tname = self.eat(lx.TYPE).value
        tag = self.eat(lx.NAME).value
        if is_keyword(tag):
            raise ParseError(f"variant tag may not be a keyword: {tag!r}", self.i)
        args = self.parse_args()
        return VariantLit(tname, tag, args)

    def parse_match(self) -> Expr:
        self.eat_kw("match")
        tname = self.eat(lx.TYPE).value  # explicit variant type (locally decodable)
        scrut = self.parse_expr()
        self.eat_kw("of")
        arms: list[MatchArm] = []
        while self.at_kw("case"):
            arms.append(self.parse_arm())
        if not arms:
            raise ParseError("match needs at least one case", self.i)
        self.eat_kw("end")
        self.eat_kw("match")
        return Match(tname, scrut, tuple(arms))

    def parse_arm(self) -> MatchArm:
        self.eat_kw("case")
        tag = self.eat(lx.NAME).value
        if is_keyword(tag):
            raise ParseError(f"variant tag may not be a keyword: {tag!r}", self.i)
        self.eat(lx.LPAREN)
        binders: list[tuple[str, Type]] = []
        if not self.at(lx.RPAREN):
            while True:
                bname = self.eat(lx.LOCAL).value
                self.eat(lx.COLON)
                bty = self.parse_type()
                binders.append((bname, bty))
                if self.at(lx.COMMA):
                    self.advance()
                    continue
                break
        self.eat(lx.RPAREN)
        self.eat(lx.ARROW)
        body = self.parse_expr()
        return MatchArm(tag, tuple(binders), body)

    def parse_cond(self) -> Expr:
        self.eat_kw("cond")
        test = self.parse_expr()
        self.eat_kw("then")
        then = self.parse_expr()
        self.eat_kw("else")
        els = self.parse_expr()
        self.eat_kw("end")
        self.eat_kw("cond")
        return Cond(test, then, els)


def parse(src: str) -> Module:
    return Parser(lx.tokenize(src)).parse_module()
