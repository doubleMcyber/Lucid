"""Type-directed AST sampler — the heart of Loom (PRD §8.1).

Given a target type and a fuel (depth) budget, the sampler builds a *well-typed*
expression by applying the typing rules in reverse: it enumerates every way to
produce a value of the requested type from in-scope bindings, declared
functions, constructors, and built-ins, then samples one. Because construction
follows the typing rules, programs are well-typed by construction, not by
rejection — the validator's type check should therefore always pass.

A whole program is assembled as: type declarations -> helper functions ->
entry function. Helpers populate the pool of callable functions and first-class
function values; the entry function is the rich one whose signature defines the
IO domain. The "no recursion / declare-before-use" rule (which, with finite
`foreach`, makes the language total) is preserved by only ever drawing from
functions generated *earlier*.

`Scope` is the scope/context manager: it tracks in-scope bindings (with
mutability) so every generated reference is valid and no binding shadows another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from lucid.ast import (
    BoolLit, BuiltinCall, Call, Cond, Expr, Field, FnDecl, FnRef, Foreach,
    IfStmt, IntLit, Let, ListLit, Match, MatchArm, Program, RecordDecl,
    RecordLit, Return, SetStmt, Stmt, StrLit, VarDecl, VarRef, VariantDecl,
    VariantLit,
)
from lucid.builtins_def import BUILTINS
from lucid.parser import Module
from lucid.types import (
    BOOL, INT, STR, RecordDef, TFn, TList, TRecord, TVariant, Type, TypeEnv,
    VariantDef, occurs_var,
)

from .config import SamplerConfig
from .rng import Rng


class SamplerError(Exception):
    """An internal invariant violation — indicates a generator bug."""


@dataclass
class FnSig:
    name: str
    params: tuple[tuple[str, Type], ...]
    ret: Type

    def fn_type(self) -> TFn:
        return TFn(tuple(t for _, t in self.params), self.ret)


@dataclass
class Scope:
    """Scope/context manager: ordered bindings with mutability, chained for
    block scoping. Shadowing is forbidden (matching the type checker)."""

    parent: Optional["Scope"] = None
    vars: list[tuple[str, Type, bool]] = field(default_factory=list)

    def child(self) -> "Scope":
        return Scope(self)

    def declare(self, name: str, ty: Type, mutable: bool) -> None:
        self.vars.append((name, ty, mutable))

    def _all(self) -> list[tuple[str, Type, bool]]:
        out: list[tuple[str, Type, bool]] = []
        s: Optional[Scope] = self
        while s is not None:
            out.extend(s.vars)
            s = s.parent
        return out

    def vars_of_type(self, ty: Type) -> list[str]:
        return [n for n, t, _ in self._all() if t == ty]

    def mutable_vars(self) -> list[tuple[str, Type]]:
        return [(n, t) for n, t, m in self._all() if m]

    def fn_values(self) -> list[tuple[str, TFn]]:
        return [(n, t) for n, t, _ in self._all() if isinstance(t, TFn)]


class Sampler:
    def __init__(self, config: Optional[SamplerConfig] = None):
        self.cfg = config or SamplerConfig()

    # ======================================================================
    # Program generation
    # ======================================================================
    def sample_program(self, rng: Rng) -> Module:
        self.rng = rng
        self.tenv = TypeEnv()
        self.record_defs: list[RecordDef] = []
        self.variant_defs: list[VariantDef] = []
        self.fn_sigs: list[FnSig] = []
        self._tc = 0  # type counter
        self._fc = 0  # field counter
        self._hc = 0  # helper counter
        self._kc = 0  # tag counter
        self._local = 0  # local-name counter (reset per function)

        decls: list = []
        decls += self._gen_type_decls()
        decls += self._gen_helpers()
        decls.append(self._gen_entry())
        return Module(Program(tuple(decls)), self.tenv)

    # -- fresh names -------------------------------------------------------
    def _fresh_local(self) -> str:
        n = self._local
        self._local += 1
        return f"v{n}"

    # ======================================================================
    # Type declarations
    # ======================================================================
    def _gen_type_decls(self) -> list:
        decls: list = []
        n_rec = self.rng.randint(0, self.cfg.max_records) if self.cfg.use_records else 0
        n_var = self.rng.randint(0, self.cfg.max_variants) if self.cfg.use_variants else 0
        # Interleave records/variants so later defs can reference earlier ones.
        plan = ["r"] * n_rec + ["v"] * n_var
        plan = self.rng.shuffle(plan)
        for kind in plan:
            if kind == "r":
                decls.append(self._gen_record_decl())
            else:
                decls.append(self._gen_variant_decl())
        return decls

    def _earlier_user_types(self) -> list[Type]:
        out: list[Type] = []
        for rd in self.record_defs:
            out.append(TRecord(rd.name))
        for vd in self.variant_defs:
            out.append(TVariant(vd.name))
        return out

    def _payload_type_pool(self) -> list[Type]:
        """Types usable as a record field / variant payload: base, earlier user
        types, and lists thereof — never the type being defined (no recursion)."""
        base = self._base_types()
        pool = list(base) + self._earlier_user_types()
        listy = [TList(t) for t in pool]
        return pool + listy

    def _gen_record_decl(self) -> RecordDecl:
        name = f"R{self._tc}"
        self._tc += 1
        pool = self._payload_type_pool()
        nf = self.rng.randint(1, self.cfg.max_fields)
        fields: list[tuple[str, Type]] = []
        for _ in range(nf):
            fname = f"f{self._fc}"
            self._fc += 1
            fields.append((fname, self.rng.choice(pool)))
        self.tenv.add_record(RecordDef(name, fields))
        self.record_defs.append(self.tenv.lookup_record(name))
        return RecordDecl(name, tuple(fields))

    def _gen_variant_decl(self) -> VariantDecl:
        name = f"V{self._tc}"
        self._tc += 1
        pool = self._payload_type_pool()
        nc = self.rng.randint(1, self.cfg.max_ctors)
        ctors: list[tuple[str, tuple[Type, ...]]] = []
        # guarantee at least one low-arity constructor for cheap leaf construction
        for i in range(nc):
            tag = f"K{self._kc}"
            self._kc += 1
            if i == 0:
                npay = self.rng.randint(0, 1)
            else:
                npay = self.rng.randint(0, self.cfg.max_payload)
            tys = tuple(self.rng.choice(pool) for _ in range(npay))
            ctors.append((tag, tys))
        self.tenv.add_variant(VariantDef(name, [(t, list(ts)) for t, ts in ctors]))
        self.variant_defs.append(self.tenv.lookup_variant(name))
        return VariantDecl(name, tuple(ctors))

    # ======================================================================
    # Type universe used for bindings / element selection
    # ======================================================================
    def _base_types(self) -> list[Type]:
        out: list[Type] = [INT]
        if self.cfg.use_bool:
            out.append(BOOL)
        if self.cfg.use_strings:
            out.append(STR)
        return out

    def _value_types(self) -> list[Type]:
        """Constructible value types (no bare function types here)."""
        base = self._base_types()
        users = self._earlier_user_types()
        out = base + users
        if self.cfg.use_lists:
            out = out + [TList(t) for t in base + users]
            if self.cfg.use_nested_lists:
                out += [TList(TList(t)) for t in base]
        return out

    def _element_types(self) -> list[Type]:
        base = self._base_types()
        out = list(base) + self._earlier_user_types()
        if self.cfg.use_lists and self.cfg.use_nested_lists:
            out += [TList(t) for t in base]
        return out

    def _binding_types(self) -> list[Type]:
        out = self._value_types()
        if self.cfg.use_higher_order:
            # function types for which a value provably exists (a helper)
            seen = set()
            for sig in self.fn_sigs:
                ft = sig.fn_type()
                if ft not in seen:
                    seen.add(ft)
                    out = out + [ft]
        return out

    # ======================================================================
    # Helper functions (the callable / first-class-value pool)
    # ======================================================================
    def _gen_helpers(self) -> list:
        decls: list = []
        n = self.rng.randint(self.cfg.min_helpers, self.cfg.max_helpers)
        for _ in range(n):
            decls.append(self._gen_helper())
        return decls

    def _gen_helper(self) -> FnDecl:
        name = f"h{self._hc}"
        self._hc += 1
        self._local = 0
        arity = self.rng.choice([1, 1, 2, 2, 0])
        ptypes_pool = self._base_types() + self._earlier_user_types()
        scope = Scope()
        params: list[tuple[str, Type]] = []
        for _ in range(arity):
            pname = self._fresh_local()
            pty = self.rng.choice(ptypes_pool)
            params.append((pname, pty))
            scope.declare(pname, pty, mutable=False)
        # Bias a fraction of helpers to be predicates ((..)->Bool) so the
        # filter intrinsic (which needs an (E)->Bool helper) is reliably
        # reachable, keeping HOF/grammar coverage high even at small scale.
        if arity >= 1 and self.cfg.use_bool and self.rng.chance(0.25):
            ret = BOOL
        else:
            ret = self.rng.choice(ptypes_pool)
        body = self._gen_block(scope, ret, self.cfg.helper_fuel, self.cfg.helper_depth)
        sig = FnSig(name, tuple(params), ret)
        decl = FnDecl(name, tuple(params), ret, body)
        self.fn_sigs.append(sig)  # available to *later* functions only
        return decl

    # ======================================================================
    # Entry function
    # ======================================================================
    def _gen_entry(self) -> FnDecl:
        name = "main"
        self._local = 0
        scope = Scope()
        nparams = self.rng.randint(self.cfg.min_params, self.cfg.max_params)
        # entry params/return are value types (no function types) so IO is clean
        ptypes = self._value_types()
        params: list[tuple[str, Type]] = []
        for _ in range(nparams):
            pname = self._fresh_local()
            pty = self.rng.choice(ptypes)
            params.append((pname, pty))
            scope.declare(pname, pty, mutable=False)
        ret = self.rng.choice(self._value_types())
        body = self._gen_block(scope, ret, self.cfg.expr_fuel, self.cfg.block_depth)
        return FnDecl(name, tuple(params), ret, body)

    # ======================================================================
    # Statement generation
    # ======================================================================
    def _gen_block(self, scope: Scope, ret: Optional[Type], fuel: int, sdepth: int) -> tuple[Stmt, ...]:
        stmts: list[Stmt] = []
        n = self.rng.randint(self.cfg.min_block_stmts, self.cfg.max_block_stmts)
        for _ in range(n):
            stmts.extend(self._gen_simple_stmt(scope, fuel, sdepth))
        if ret is not None:
            stmts.append(self._gen_terminator(ret, scope, fuel, sdepth))
        return tuple(stmts)

    def _gen_simple_stmt(self, scope: Scope, fuel: int, sdepth: int) -> list[Stmt]:
        options: list[tuple[float, object]] = [(self.cfg.w_bind, "bind")]
        if scope.mutable_vars():
            options.append((self.cfg.w_set, "set"))
        if self.cfg.use_foreach and self.cfg.use_lists and sdepth > 0:
            options.append((self.cfg.w_foreach, "foreach"))
        if self.cfg.use_if and sdepth > 0:
            options.append((self.cfg.w_if, "if"))
        kind = self.rng.weighted([(w, (lambda k=k: k)) for w, k in options])

        if kind == "bind":
            return [self._gen_binding(scope, fuel)]
        if kind == "set":
            return [self._gen_set(scope, fuel)]
        if kind == "foreach":
            return self._gen_accumulator_foreach(scope, fuel, sdepth)
        if kind == "if":
            return [self._gen_if_nonret(scope, fuel, sdepth)]
        raise SamplerError(f"unknown stmt kind {kind}")

    def _gen_binding(self, scope: Scope, fuel: int) -> Stmt:
        ty = self.rng.choice(self._binding_types())
        name = self._fresh_local()
        expr = self._gen_expr(ty, scope, fuel)
        mutable = self.rng.chance(self.cfg.p_mutable)
        scope.declare(name, ty, mutable)
        return VarDecl(name, ty, expr) if mutable else Let(name, ty, expr)

    def _gen_set(self, scope: Scope, fuel: int) -> Stmt:
        name, ty = self.rng.choice(scope.mutable_vars())
        return SetStmt(name, self._gen_expr(ty, scope, fuel))

    def _gen_accumulator_foreach(self, scope: Scope, fuel: int, sdepth: int) -> list[Stmt]:
        acc_ty = self.rng.choice(self._base_types())
        acc = self._fresh_local()
        init = self._gen_expr(acc_ty, scope, max(1, fuel - 1))
        decl = VarDecl(acc, acc_ty, init)
        scope.declare(acc, acc_ty, mutable=True)

        elem_ty = self.rng.choice(self._element_types())
        it = self._gen_expr(TList(elem_ty), scope, fuel)
        loopvar = self._fresh_local()
        inner = scope.child()
        inner.declare(loopvar, elem_ty, mutable=False)

        body: list[Stmt] = []
        extra = self.rng.randint(0, self.cfg.max_loop_stmts)
        for _ in range(extra):
            body.extend(self._gen_simple_stmt(inner, fuel, sdepth - 1))
        body.append(SetStmt(acc, self._gen_expr(acc_ty, inner, fuel)))
        return [decl, Foreach(loopvar, elem_ty, it, tuple(body))]

    def _gen_if_nonret(self, scope: Scope, fuel: int, sdepth: int) -> Stmt:
        cond = self._gen_expr(BOOL, scope, fuel)
        then_body = self._gen_block(scope.child(), None, fuel, sdepth - 1)
        else_body = self._gen_block(scope.child(), None, fuel, sdepth - 1)
        return IfStmt(cond, then_body, else_body)

    def _gen_terminator(self, ret: Type, scope: Scope, fuel: int, sdepth: int) -> Stmt:
        if self.cfg.use_if and sdepth > 0 and self.rng.chance(self.cfg.p_if_return):
            cond = self._gen_expr(BOOL, scope, fuel)
            then_body = self._gen_block(scope.child(), ret, fuel, sdepth - 1)
            else_body = self._gen_block(scope.child(), ret, fuel, sdepth - 1)
            return IfStmt(cond, then_body, else_body)
        return Return(self._gen_expr(ret, scope, fuel))

    # ======================================================================
    # Expression generation (type-directed)
    # ======================================================================
    def _gen_expr(self, ty: Type, scope: Scope, fuel: int) -> Expr:
        cands: list[tuple[float, object]] = []
        # leaves: variable references
        for v in scope.vars_of_type(ty):
            cands.append((self.cfg.w_var, (lambda v=v: VarRef(v))))
        # leaves: literals
        if ty == INT:
            cands.append((self.cfg.w_lit, lambda: IntLit(self.rng.randint(self.cfg.int_min, self.cfg.int_max))))
        elif ty == BOOL:
            cands.append((self.cfg.w_lit, lambda: BoolLit(self.rng.chance(0.5))))
        elif ty == STR and self.cfg.use_strings:
            cands.append((self.cfg.w_lit, lambda: StrLit(self.rng.choice(self.cfg.words))))

        if fuel > 0:
            self._add_compound(cands, ty, scope, fuel)

        if not cands:
            return self._gen_leaf(ty, scope)
        return self.rng.weighted(cands)

    def _gen_leaf(self, ty: Type, scope: Scope) -> Expr:
        vs = scope.vars_of_type(ty)
        if vs and self.rng.chance(0.7):
            return VarRef(self.rng.choice(vs))
        if ty == INT:
            return IntLit(self.rng.randint(self.cfg.int_min, self.cfg.int_max))
        if ty == BOOL:
            return BoolLit(self.rng.chance(0.5))
        if ty == STR:
            return StrLit(self.rng.choice(self.cfg.words))
        if isinstance(ty, TList):
            return ListLit(ty.elem, ())  # empty list — always valid
        if isinstance(ty, TRecord):
            rd = self.tenv.lookup_record(ty.name)
            fields = tuple((n, self._gen_leaf(ft, scope)) for n, ft in rd.fields)
            return RecordLit(ty.name, tuple(sorted(fields, key=lambda nv: nv[0])))
        if isinstance(ty, TVariant):
            vd = self.tenv.lookup_variant(ty.name)
            tag, tys = min(vd.ctors, key=lambda c: len(c[1]))
            return VariantLit(ty.name, tag, tuple(self._gen_leaf(t, scope) for t in tys))
        if isinstance(ty, TFn):
            if vs:
                return VarRef(self.rng.choice(vs))
            fn = self._fn_value_for(ty)
            if fn is not None:
                return FnRef(fn)
            raise SamplerError(f"no leaf for function type {ty}")
        raise SamplerError(f"no leaf for type {ty}")

    # -- compound producers ------------------------------------------------
    def _add_compound(self, cands: list, ty: Type, scope: Scope, fuel: int) -> None:
        f = fuel - 1
        # user-function calls returning ty
        for sig in self.fn_sigs:
            if sig.ret == ty:
                cands.append((self.cfg.w_call, self._mk_call(sig, scope, f)))
        # record field projections yielding ty
        for rd in self.record_defs:
            for fname, fty in rd.fields:
                if fty == ty:
                    cands.append((self.cfg.w_field, self._mk_field(rd.name, fname, scope, f)))
        # cond
        if self.cfg.use_cond:
            cands.append((self.cfg.w_cond, self._mk_cond(ty, scope, f)))
        # match (ty as the arm-result type)
        if self.cfg.use_match and self.variant_defs:
            vd = self.rng.choice(self.variant_defs)
            cands.append((self.cfg.w_match, self._mk_match(vd, ty, scope, f)))
        # built-ins
        self._add_builtin_producers(cands, ty, scope, f)
        # constructors
        if isinstance(ty, TList):
            cands.append((self.cfg.w_listlit, self._mk_listlit(ty.elem, scope, f)))
        if isinstance(ty, TRecord):
            cands.append((self.cfg.w_record, self._mk_record(ty.name, scope, f)))
        if isinstance(ty, TVariant):
            vd = self.tenv.lookup_variant(ty.name)
            tag, tys = self.rng.choice(vd.ctors)
            cands.append((self.cfg.w_variant, self._mk_variant(ty.name, tag, tys, scope, f)))
        # first-class function value (FnRef) for function types
        if isinstance(ty, TFn):
            fn = self._fn_value_for(ty)
            if fn is not None:
                cands.append((self.cfg.w_var, (lambda fn=fn: FnRef(fn))))

    def _mk_call(self, sig: FnSig, scope: Scope, f: int):
        def thunk():
            args = tuple(self._gen_expr(pt, scope, f) for _, pt in sig.params)
            return Call(sig.name, args)
        return thunk

    def _mk_field(self, rname: str, fname: str, scope: Scope, f: int):
        def thunk():
            return Field(fname, self._gen_expr(TRecord(rname), scope, f))
        return thunk

    def _mk_cond(self, ty: Type, scope: Scope, f: int):
        def thunk():
            return Cond(
                self._gen_expr(BOOL, scope, f),
                self._gen_expr(ty, scope, f),
                self._gen_expr(ty, scope, f),
            )
        return thunk

    def _mk_match(self, vd: VariantDef, ty: Type, scope: Scope, f: int):
        def thunk():
            scrut = self._gen_expr(TVariant(vd.name), scope, f)
            arms: list[MatchArm] = []
            for tag, tys in vd.ctors:
                inner = scope.child()
                binders: list[tuple[str, Type]] = []
                for t in tys:
                    bn = self._fresh_local()
                    inner.declare(bn, t, mutable=False)
                    binders.append((bn, t))
                body = self._gen_expr(ty, inner, f)
                arms.append(MatchArm(tag, tuple(binders), body))
            return Match(vd.name, scrut, tuple(arms))
        return thunk

    def _mk_listlit(self, elem: Type, scope: Scope, f: int):
        def thunk():
            k = self.rng.randint(1, self.cfg.max_list_lit)
            return ListLit(elem, tuple(self._gen_expr(elem, scope, f) for _ in range(k)))
        return thunk

    def _mk_record(self, rname: str, scope: Scope, f: int):
        rd = self.tenv.lookup_record(rname)

        def thunk():
            fields = tuple((n, self._gen_expr(ft, scope, f)) for n, ft in rd.fields)
            return RecordLit(rname, tuple(sorted(fields, key=lambda nv: nv[0])))
        return thunk

    def _mk_variant(self, vname: str, tag: str, tys, scope: Scope, f: int):
        def thunk():
            return VariantLit(vname, tag, tuple(self._gen_expr(t, scope, f) for t in tys))
        return thunk

    # -- built-in producers (type-directed, with intrinsic resolution) -----
    def _fn_values(self, scope: Scope) -> list[tuple[Expr, TFn]]:
        out: list[tuple[Expr, TFn]] = []
        for sig in self.fn_sigs:
            out.append((FnRef(sig.name), sig.fn_type()))
        for n, ft in scope.fn_values():
            out.append((VarRef(n), ft))
        return out

    def _fn_value_for(self, ty: TFn) -> Optional[str]:
        matches = [sig.name for sig in self.fn_sigs if sig.fn_type() == ty]
        if matches:
            return self.rng.choice(matches)
        return None

    def _type_enabled(self, t: Type) -> bool:
        """Whether a type is permitted under the current feature flags. Used to
        keep disabled types from sneaking in as built-in arguments (e.g. a
        string-consuming built-in when use_strings is off)."""
        if t == STR:
            return self.cfg.use_strings
        if t == BOOL:
            return self.cfg.use_bool
        if t == INT:
            return True
        if isinstance(t, TList):
            return self.cfg.use_lists and self._type_enabled(t.elem)
        if isinstance(t, TFn):
            return (self.cfg.use_higher_order
                    and all(self._type_enabled(p) for p in t.params)
                    and self._type_enabled(t.ret))
        if isinstance(t, TRecord):
            return self.cfg.use_records and self.tenv.lookup_record(t.name) is not None
        if isinstance(t, TVariant):
            return self.cfg.use_variants and self.tenv.lookup_variant(t.name) is not None
        return True

    def _add_builtin_producers(self, cands: list, ty: Type, scope: Scope, f: int) -> None:
        # monomorphic built-ins whose return type is exactly ty
        for b in BUILTINS.values():
            if occurs_var(b.ret) or any(occurs_var(p) for p in b.params):
                continue
            if not self._type_enabled(b.ret) or not all(self._type_enabled(p) for p in b.params):
                continue
            if b.ret == ty:
                cands.append((self.cfg.w_builtin, self._mk_builtin(b.name, b.params, scope, f)))

        if not self.cfg.use_lists:
            self._add_hof_producers(cands, ty, scope, f)
            return

        elem_pool = self._element_types()

        # polymorphic intrinsics, specialized to the requested type
        if ty == INT:
            cands.append((self.cfg.w_builtin, self._mk_builtin1("length", lambda: TList(self.rng.choice(elem_pool)), scope, f)))
        if ty == BOOL and self.cfg.use_bool:
            cands.append((self.cfg.w_builtin, self._mk_builtin1("is_empty", lambda: TList(self.rng.choice(elem_pool)), scope, f)))

        if isinstance(ty, TList):
            E = ty.elem
            cands.append((self.cfg.w_builtin, self._mk_builtin("append", (ty, E), scope, f)))
            cands.append((self.cfg.w_builtin, self._mk_builtin("concat_list", (ty, ty), scope, f)))
            cands.append((self.cfg.w_builtin, self._mk_builtin("reverse", (ty,), scope, f)))
            if E == INT:
                cands.append((self.cfg.w_builtin, self._mk_builtin("range", (INT, INT), scope, f)))

        # get_or / head_or produce any value type. When nested lists are
        # disabled, skip these for list-typed results (their List[ty] argument
        # would be a nested list), honoring the use_nested_lists ablation.
        if self._is_value_type(ty) and (self.cfg.use_nested_lists or not isinstance(ty, TList)):
            cands.append((self.cfg.w_builtin, self._mk_builtin("get_or", (TList(ty), INT, ty), scope, f)))
            cands.append((self.cfg.w_builtin, self._mk_builtin("head_or", (TList(ty), ty), scope, f)))

        self._add_hof_producers(cands, ty, scope, f)

    def _add_hof_producers(self, cands: list, ty: Type, scope: Scope, f: int) -> None:

        if not self.cfg.use_higher_order:
            return

        fnvals = self._fn_values(scope)

        # map : produce List[E2] using a helper (E1)->E2
        if isinstance(ty, TList):
            e2 = ty.elem
            for fexpr, ft in fnvals:
                if len(ft.params) == 1 and ft.ret == e2:
                    e1 = ft.params[0]
                    cands.append((self.cfg.w_hof, self._mk_map(e1, fexpr, scope, f)))
                if len(ft.params) == 1 and ft.params[0] == ty.elem and ft.ret == BOOL:
                    cands.append((self.cfg.w_hof, self._mk_filter(ty.elem, fexpr, scope, f)))

        # foldl : produce ty using a helper (ty, E)->ty (needs a list to fold)
        for fexpr, ft in fnvals:
            if self.cfg.use_lists and len(ft.params) == 2 and ft.ret == ty and ft.params[0] == ty:
                e = ft.params[1]
                cands.append((self.cfg.w_hof, self._mk_foldl(e, ty, fexpr, scope, f)))
            if len(ft.params) == 1 and ft.ret == ty:
                cands.append((self.cfg.w_hof, self._mk_apply1(ft.params[0], fexpr, scope, f)))
            if len(ft.params) == 2 and ft.ret == ty:
                cands.append((self.cfg.w_hof, self._mk_apply2(ft.params[0], ft.params[1], fexpr, scope, f)))

    def _is_value_type(self, ty: Type) -> bool:
        return isinstance(ty, (TList, TRecord, TVariant)) or ty in (INT, BOOL, STR)

    def _mk_builtin(self, name: str, ptypes, scope: Scope, f: int):
        def thunk():
            return BuiltinCall(name, tuple(self._gen_expr(pt, scope, f) for pt in ptypes))
        return thunk

    def _mk_builtin1(self, name: str, ptype_fn, scope: Scope, f: int):
        def thunk():
            return BuiltinCall(name, (self._gen_expr(ptype_fn(), scope, f),))
        return thunk

    def _mk_map(self, e1: Type, fexpr: Expr, scope: Scope, f: int):
        def thunk():
            return BuiltinCall("map", (self._gen_expr(TList(e1), scope, f), fexpr))
        return thunk

    def _mk_filter(self, e: Type, fexpr: Expr, scope: Scope, f: int):
        def thunk():
            return BuiltinCall("filter", (self._gen_expr(TList(e), scope, f), fexpr))
        return thunk

    def _mk_foldl(self, e: Type, ty: Type, fexpr: Expr, scope: Scope, f: int):
        def thunk():
            return BuiltinCall(
                "foldl",
                (self._gen_expr(TList(e), scope, f), self._gen_expr(ty, scope, f), fexpr),
            )
        return thunk

    def _mk_apply1(self, e: Type, fexpr: Expr, scope: Scope, f: int):
        def thunk():
            return BuiltinCall("apply1", (fexpr, self._gen_expr(e, scope, f)))
        return thunk

    def _mk_apply2(self, e1: Type, e2: Type, fexpr: Expr, scope: Scope, f: int):
        def thunk():
            return BuiltinCall(
                "apply2",
                (fexpr, self._gen_expr(e1, scope, f), self._gen_expr(e2, scope, f)),
            )
        return thunk
