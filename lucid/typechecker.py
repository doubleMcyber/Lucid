"""Lucid type checker (PRD §7.4).

Static, strong, monomorphic: no implicit coercion, no subtyping, no user-facing
overloading. The only polymorphism is in the built-in list intrinsics, resolved
by a one-shot structural matcher (`_unify`).

It also enforces the properties that make the language *total* (PRD §7.5):
  * No recursion / forward references: a function or `@f` value may only refer to
    functions declared earlier, so the call graph is a DAG. Combined with the
    fact that the only iteration (`foreach`) ranges over a finite list, every
    program provably terminates.
  * Guaranteed return: every function body must return on all paths, with the
    declared type, and no statement may follow a terminator (no dead code).

A passing type check is the validator's contract: if a sampler-built program
fails here, that is a *generator bug*, logged loudly by Loom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .ast import (
    BoolLit, BuiltinCall, Call, Cond, Expr, Field, FnDecl, FnRef, Foreach,
    IfStmt, IntLit, Let, ListLit, Match, Program, RecordDecl, RecordLit,
    Return, SetStmt, Stmt, StrLit, VarDecl, VarRef, VariantDecl, VariantLit,
)
from .builtins_def import BUILTINS
from .errors import TypeError_
from .parser import Module
from .types import (
    BOOL, INT, STR, TFn, TList, TRecord, TVar, TVariant, Type, TypeEnv,
)


@dataclass
class _Scope:
    parent: Optional["_Scope"] = None

    def __post_init__(self):
        self.vars: dict[str, tuple[Type, bool]] = {}

    def child(self) -> "_Scope":
        return _Scope(self)

    def visible(self, name: str) -> bool:
        s: Optional[_Scope] = self
        while s is not None:
            if name in s.vars:
                return True
            s = s.parent
        return False

    def lookup(self, name: str) -> Optional[tuple[Type, bool]]:
        s: Optional[_Scope] = self
        while s is not None:
            if name in s.vars:
                return s.vars[name]
            s = s.parent
        return None

    def declare(self, name: str, ty: Type, mutable: bool) -> None:
        if self.visible(name):
            raise TypeError_(f"local ${name} shadows an existing binding")
        self.vars[name] = (ty, mutable)


class TypeChecker:
    def __init__(self, module: Module):
        self.tenv: TypeEnv = module.tenv
        self.program: Program = module.program
        self.fns: dict[str, FnDecl] = {}  # functions declared *so far*

    # -- public ------------------------------------------------------------
    def check(self) -> None:
        for d in self.program.decls:
            if isinstance(d, RecordDecl):
                self._check_record(d)
            elif isinstance(d, VariantDecl):
                self._check_variant(d)
            elif isinstance(d, FnDecl):
                self._check_fn(d)
                self.fns[d.name] = d

    # -- declarations ------------------------------------------------------
    def _check_record(self, d: RecordDecl) -> None:
        seen = set()
        for n, t in d.fields:
            if n in seen:
                raise TypeError_(f"duplicate field %{n} in record #{d.name}")
            seen.add(n)
            self._wf(t)

    def _check_variant(self, d: VariantDecl) -> None:
        if not d.ctors:
            raise TypeError_(f"variant #{d.name} must have at least one constructor")
        seen = set()
        for tag, tys in d.ctors:
            if tag in seen:
                raise TypeError_(f"duplicate tag {tag} in variant #{d.name}")
            seen.add(tag)
            for t in tys:
                self._wf(t)

    def _check_fn(self, fn: FnDecl) -> None:
        if fn.name in self.fns:
            raise TypeError_(f"duplicate function @{fn.name}")
        scope = _Scope()
        pseen = set()
        for pname, pty in fn.params:
            if pname in pseen:
                raise TypeError_(f"duplicate parameter ${pname} in @{fn.name}")
            pseen.add(pname)
            self._wf(pty)
            scope.declare(pname, pty, mutable=False)
        self._wf(fn.ret)
        terminates = self._check_block(fn.body, scope, fn.ret)
        if not terminates:
            raise TypeError_(f"@{fn.name} may finish without returning a #{fn.ret}")

    # -- well-formedness of types -----------------------------------------
    def _wf(self, t: Type) -> None:
        if isinstance(t, TList):
            self._wf(t.elem)
        elif isinstance(t, TFn):
            for p in t.params:
                self._wf(p)
            self._wf(t.ret)
        elif isinstance(t, TRecord):
            if self.tenv.lookup_record(t.name) is None:
                raise TypeError_(f"unknown record type #{t.name}")
        elif isinstance(t, TVariant):
            if self.tenv.lookup_variant(t.name) is None:
                raise TypeError_(f"unknown variant type #{t.name}")
        elif isinstance(t, TVar):
            raise TypeError_("type variables are not allowed in surface types")

    # -- statements (return bool: does this always return?) ----------------
    def _check_block(self, stmts: tuple[Stmt, ...], scope: _Scope, ret: Type) -> bool:
        terminated = False
        for s in stmts:
            if terminated:
                raise TypeError_("unreachable statement after a return")
            terminated = self._check_stmt(s, scope, ret)
        return terminated

    def _check_stmt(self, s: Stmt, scope: _Scope, ret: Type) -> bool:
        if isinstance(s, Let):
            self._wf(s.ty)
            self._expect(s.expr, s.ty, scope, f"binding ${s.name}")
            scope.declare(s.name, s.ty, mutable=False)
            return False
        if isinstance(s, VarDecl):
            self._wf(s.ty)
            self._expect(s.expr, s.ty, scope, f"binding ${s.name}")
            scope.declare(s.name, s.ty, mutable=True)
            return False
        if isinstance(s, SetStmt):
            info = scope.lookup(s.name)
            if info is None:
                raise TypeError_(f"set of undeclared local ${s.name}")
            ty, mutable = info
            if not mutable:
                raise TypeError_(f"set of immutable ${s.name} (declared with let/param)")
            self._expect(s.expr, ty, scope, f"set ${s.name}")
            return False
        if isinstance(s, Foreach):
            self._wf(s.elem_ty)
            it = self._infer(s.iter, scope)
            if it != TList(s.elem_ty):
                raise TypeError_(
                    f"foreach expects #List[{s.elem_ty}], got {it}"
                )
            inner = scope.child()
            inner.declare(s.var, s.elem_ty, mutable=False)
            self._check_block(s.body, inner, ret)
            return False  # loop may execute zero times
        if isinstance(s, IfStmt):
            self._expect(s.cond, BOOL, scope, "if condition")
            t1 = self._check_block(s.then_body, scope.child(), ret)
            t2 = self._check_block(s.else_body, scope.child(), ret)
            return t1 and t2
        if isinstance(s, Return):
            self._expect(s.expr, ret, scope, "return value")
            return True
        raise TypeError_(f"unknown statement {type(s).__name__}")

    # -- expressions -------------------------------------------------------
    def _expect(self, e: Expr, ty: Type, scope: _Scope, ctx: str) -> None:
        got = self._infer(e, scope)
        if got != ty:
            raise TypeError_(f"{ctx}: expected {ty}, got {got}")

    def _infer(self, e: Expr, scope: _Scope) -> Type:
        if isinstance(e, IntLit):
            if e.value < 0:
                raise TypeError_("integer literals must be non-negative")
            return INT
        if isinstance(e, BoolLit):
            return BOOL
        if isinstance(e, StrLit):
            return STR
        if isinstance(e, VarRef):
            info = scope.lookup(e.name)
            if info is None:
                raise TypeError_(f"reference to undeclared local ${e.name}")
            return info[0]
        if isinstance(e, FnRef):
            fn = self.fns.get(e.name)
            if fn is None:
                raise TypeError_(
                    f"@{e.name} is not a function declared earlier (no recursion)"
                )
            return TFn(tuple(t for _, t in fn.params), fn.ret)
        if isinstance(e, BuiltinCall):
            return self._infer_builtin(e, scope)
        if isinstance(e, Call):
            return self._infer_call(e, scope)
        if isinstance(e, ListLit):
            self._wf(e.elem_ty)
            for el in e.elems:
                self._expect(el, e.elem_ty, scope, "list element")
            return TList(e.elem_ty)
        if isinstance(e, RecordLit):
            return self._infer_record(e, scope)
        if isinstance(e, Field):
            return self._infer_field(e, scope)
        if isinstance(e, VariantLit):
            return self._infer_variant(e, scope)
        if isinstance(e, Match):
            return self._infer_match(e, scope)
        if isinstance(e, Cond):
            self._expect(e.test, BOOL, scope, "cond test")
            t_then = self._infer(e.then, scope)
            t_els = self._infer(e.els, scope)
            if t_then != t_els:
                raise TypeError_(
                    f"cond branches differ: {t_then} vs {t_els}"
                )
            return t_then
        raise TypeError_(f"unknown expression {type(e).__name__}")

    def _infer_builtin(self, e: BuiltinCall, scope: _Scope) -> Type:
        b = BUILTINS.get(e.name)
        if b is None:
            raise TypeError_(f"unknown built-in {e.name}")
        if len(e.args) != b.arity:
            raise TypeError_(
                f"{e.name} expects {b.arity} args, got {len(e.args)}"
            )
        arg_types = [self._infer(a, scope) for a in e.args]
        subst: dict[str, Type] = {}
        for pty, aty in zip(b.params, arg_types):
            self._unify(pty, aty, subst, e.name)
        return self._subst(b.ret, subst)

    def _infer_call(self, e: Call, scope: _Scope) -> Type:
        fn = self.fns.get(e.name)
        if fn is None:
            raise TypeError_(
                f"call to @{e.name}: not declared earlier (no recursion in v1)"
            )
        if len(e.args) != len(fn.params):
            raise TypeError_(
                f"@{e.name} expects {len(fn.params)} args, got {len(e.args)}"
            )
        for arg, (_, pty) in zip(e.args, fn.params):
            self._expect(arg, pty, scope, f"argument to @{e.name}")
        return fn.ret

    def _infer_record(self, e: RecordLit, scope: _Scope) -> Type:
        rd = self.tenv.lookup_record(e.type_name)
        if rd is None:
            raise TypeError_(f"unknown record type #{e.type_name}")
        declared = set(rd.field_names())
        got = [n for n, _ in e.fields]
        # Field order is normalized by the canonicalizer, so accept any order;
        # require exactly the declared field set with no duplicates.
        if len(got) != len(set(got)):
            raise TypeError_(f"#{e.type_name} has a duplicate field in {got}")
        if set(got) != declared:
            raise TypeError_(
                f"#{e.type_name} fields must be {sorted(declared)}, got {sorted(got)}"
            )
        for (n, fe) in e.fields:
            self._expect(fe, rd.field_type(n), scope, f"field %{n}")
        return TRecord(e.type_name)

    def _infer_field(self, e: Field, scope: _Scope) -> Type:
        rt = self._infer(e.record, scope)
        if not isinstance(rt, TRecord):
            raise TypeError_(f"get %{e.field} on non-record {rt}")
        rd = self.tenv.lookup_record(rt.name)
        ft = rd.field_type(e.field) if rd else None
        if ft is None:
            raise TypeError_(f"record #{rt.name} has no field %{e.field}")
        return ft

    def _infer_variant(self, e: VariantLit, scope: _Scope) -> Type:
        vd = self.tenv.lookup_variant(e.type_name)
        if vd is None:
            raise TypeError_(f"unknown variant type #{e.type_name}")
        ctys = vd.ctor_types(e.tag)
        if ctys is None:
            raise TypeError_(f"variant #{e.type_name} has no tag {e.tag}")
        if len(e.args) != len(ctys):
            raise TypeError_(
                f"#{e.type_name}.{e.tag} expects {len(ctys)} args, got {len(e.args)}"
            )
        for arg, ty in zip(e.args, ctys):
            self._expect(arg, ty, scope, f"#{e.type_name}.{e.tag} payload")
        return TVariant(e.type_name)

    def _infer_match(self, e: Match, scope: _Scope) -> Type:
        vd = self.tenv.lookup_variant(e.type_name)
        if vd is None:
            raise TypeError_(f"unknown variant type #{e.type_name}")
        st = self._infer(e.scrutinee, scope)
        if st != TVariant(e.type_name):
            raise TypeError_(f"match on #{e.type_name} but scrutinee is {st}")
        tags_in_arms = [a.tag for a in e.arms]
        if len(set(tags_in_arms)) != len(tags_in_arms):
            raise TypeError_(f"duplicate case in match on #{e.type_name}")
        if set(tags_in_arms) != set(vd.tag_names()):
            raise TypeError_(
                f"match on #{e.type_name} must be exhaustive: "
                f"cover {vd.tag_names()}, got {tags_in_arms}"
            )
        result_ty: Optional[Type] = None
        for arm in e.arms:
            ctys = vd.ctor_types(arm.tag)
            if len(arm.binders) != len(ctys):
                raise TypeError_(
                    f"case {arm.tag} binds {len(arm.binders)} of {len(ctys)} payloads"
                )
            arm_scope = scope.child()
            for (bname, bty), pty in zip(arm.binders, ctys):
                if bty != pty:
                    raise TypeError_(
                        f"case {arm.tag} binder ${bname}: declared {bty}, payload {pty}"
                    )
                arm_scope.declare(bname, bty, mutable=False)
            bt = self._infer(arm.body, arm_scope)
            if result_ty is None:
                result_ty = bt
            elif bt != result_ty:
                raise TypeError_(
                    f"match arms differ: {result_ty} vs {bt}"
                )
        assert result_ty is not None
        return result_ty

    # -- intrinsic polymorphism matcher -----------------------------------
    def _unify(self, pat: Type, actual: Type, subst: dict[str, Type], who: str) -> None:
        if isinstance(pat, TVar):
            bound = subst.get(pat.name)
            if bound is None:
                subst[pat.name] = actual
            elif bound != actual:
                raise TypeError_(
                    f"{who}: type variable #{pat.name} = {bound} vs {actual}"
                )
            return
        if isinstance(pat, TList):
            if not isinstance(actual, TList):
                raise TypeError_(f"{who}: expected a list, got {actual}")
            self._unify(pat.elem, actual.elem, subst, who)
            return
        if isinstance(pat, TFn):
            if not isinstance(actual, TFn) or len(pat.params) != len(actual.params):
                raise TypeError_(f"{who}: expected {pat}, got {actual}")
            for pp, ap in zip(pat.params, actual.params):
                self._unify(pp, ap, subst, who)
            self._unify(pat.ret, actual.ret, subst, who)
            return
        if pat != actual:
            raise TypeError_(f"{who}: expected {pat}, got {actual}")

    def _subst(self, t: Type, subst: dict[str, Type]) -> Type:
        if isinstance(t, TVar):
            if t.name not in subst:
                raise TypeError_(f"unresolved type variable #{t.name}")
            return subst[t.name]
        if isinstance(t, TList):
            return TList(self._subst(t.elem, subst))
        if isinstance(t, TFn):
            return TFn(
                tuple(self._subst(p, subst) for p in t.params),
                self._subst(t.ret, subst),
            )
        return t


def typecheck(module: Module) -> None:
    """Raise TypeError_ if the module is ill-typed; return None on success."""
    TypeChecker(module).check()
