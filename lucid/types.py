"""Lucid type representations.

Types are immutable, hashable value objects so they can be used as dict keys
(the sampler indexes "ways to build a value of type T" by type) and compared
structurally. Records and variants are *nominal*: equality is by name, and the
shape lives in a `TypeEnv` registry that the parser/typechecker/sampler share.

Type variables (`TVar`) appear only inside built-in intrinsic signatures
(e.g. `length : (#List[#a]) -> #Int`). They are resolved by a tiny one-shot
matcher in the typechecker; Lucid v1 has no user-facing generics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class Type:
    """Base class for all Lucid types. Subclasses are frozen dataclasses."""

    def __str__(self) -> str:  # pragma: no cover - overridden everywhere
        raise NotImplementedError


@dataclass(frozen=True)
class TInt(Type):
    def __str__(self) -> str:
        return "#Int"


@dataclass(frozen=True)
class TBool(Type):
    def __str__(self) -> str:
        return "#Bool"


@dataclass(frozen=True)
class TStr(Type):
    def __str__(self) -> str:
        return "#Str"


@dataclass(frozen=True)
class TList(Type):
    elem: Type

    def __str__(self) -> str:
        return f"#List[{self.elem}]"


@dataclass(frozen=True)
class TFn(Type):
    params: tuple[Type, ...]
    ret: Type

    def __str__(self) -> str:
        ps = ", ".join(str(p) for p in self.params)
        return f"#Fn[({ps}) -> {self.ret}]"


@dataclass(frozen=True)
class TRecord(Type):
    """Nominal record type; field layout is stored in the TypeEnv."""

    name: str

    def __str__(self) -> str:
        return f"#{self.name}"


@dataclass(frozen=True)
class TVariant(Type):
    """Nominal tagged-variant type; constructors stored in the TypeEnv."""

    name: str

    def __str__(self) -> str:
        return f"#{self.name}"


@dataclass(frozen=True)
class TVar(Type):
    """Intrinsic-only type variable. Not part of the surface language."""

    name: str

    def __str__(self) -> str:
        return f"#{self.name}"


# Singletons for the base types (cheap identity, but value-equal too).
INT = TInt()
BOOL = TBool()
STR = TStr()

BASE_TYPES: tuple[Type, ...] = (INT, BOOL, STR)


@dataclass
class RecordDef:
    name: str
    # Ordered list of (field_name, field_type); order is the canonical order.
    fields: list[tuple[str, Type]]

    def field_type(self, fname: str) -> Optional[Type]:
        for n, t in self.fields:
            if n == fname:
                return t
        return None

    def field_names(self) -> list[str]:
        return [n for n, _ in self.fields]


@dataclass
class VariantDef:
    name: str
    # Ordered list of (tag_name, [payload_types]); order is canonical.
    ctors: list[tuple[str, list[Type]]]

    def ctor_types(self, tag: str) -> Optional[list[Type]]:
        for t, ts in self.ctors:
            if t == tag:
                return ts
        return None

    def tag_names(self) -> list[str]:
        return [t for t, _ in self.ctors]


@dataclass
class TypeEnv:
    """Registry of user-declared record and variant types for one program."""

    records: dict[str, RecordDef] = field(default_factory=dict)
    variants: dict[str, VariantDef] = field(default_factory=dict)

    def add_record(self, rd: RecordDef) -> None:
        if rd.name in self.records or rd.name in self.variants:
            raise ValueError(f"duplicate type name #{rd.name}")
        self.records[rd.name] = rd

    def add_variant(self, vd: VariantDef) -> None:
        if vd.name in self.records or vd.name in self.variants:
            raise ValueError(f"duplicate type name #{vd.name}")
        self.variants[vd.name] = vd

    def is_declared(self, name: str) -> bool:
        return name in self.records or name in self.variants

    def lookup_record(self, name: str) -> Optional[RecordDef]:
        return self.records.get(name)

    def lookup_variant(self, name: str) -> Optional[VariantDef]:
        return self.variants.get(name)


def is_base(t: Type) -> bool:
    return isinstance(t, (TInt, TBool, TStr))


def occurs_var(t: Type) -> bool:
    """True if a type mentions an intrinsic TVar (used only by the matcher)."""
    if isinstance(t, TVar):
        return True
    if isinstance(t, TList):
        return occurs_var(t.elem)
    if isinstance(t, TFn):
        return any(occurs_var(p) for p in t.params) or occurs_var(t.ret)
    return False
