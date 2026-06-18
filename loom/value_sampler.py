"""Type-directed sampling of *runtime values* (for IO inputs).

The AST sampler builds programs; this builds the inputs we feed them. Given an
entry signature, it samples concrete Python values of each parameter type so the
interpreter can produce (input, output) pairs (PRD §8.3 IO examples).

Like everything in Loom it is fully driven by a seeded `Rng`, so the inputs —
and therefore the IO pairs and the dataset bytes — are reproducible.
"""

from __future__ import annotations

from typing import Any, Optional

from lucid.types import (
    BOOL, INT, STR, TFn, TList, TRecord, TVariant, Type, TypeEnv,
)
from lucid.values import FnVal, RecordVal, VariantVal

from .rng import Rng


class ValueSampler:
    def __init__(self, tenv: TypeEnv, rng: Rng, fn_names_by_type: Optional[dict] = None,
                 int_min: int = -20, int_max: int = 20, max_list: int = 5,
                 words: Optional[list[str]] = None, max_depth: int = 4):
        self.tenv = tenv
        self.rng = rng
        self.fn_names_by_type = fn_names_by_type or {}
        self.int_min = int_min
        self.int_max = int_max
        self.max_list = max_list
        self.words = words or ["", "a", "ab", "abc", "x", "foo", "bar", "hello"]
        self.max_depth = max_depth

    def sample(self, ty: Type, depth: int = 0) -> Any:
        if ty == INT:
            return self.rng.randint(self.int_min, self.int_max)
        if ty == BOOL:
            return self.rng.chance(0.5)
        if ty == STR:
            return self.rng.choice(self.words)
        if isinstance(ty, TList):
            n = 0 if depth >= self.max_depth else self.rng.randint(0, self.max_list)
            return [self.sample(ty.elem, depth + 1) for _ in range(n)]
        if isinstance(ty, TRecord):
            rd = self.tenv.lookup_record(ty.name)
            return RecordVal(
                ty.name,
                tuple((n, self.sample(ft, depth + 1)) for n, ft in rd.fields),
            )
        if isinstance(ty, TVariant):
            vd = self.tenv.lookup_variant(ty.name)
            # at depth limit, prefer the lowest-arity constructor to stay finite
            if depth >= self.max_depth:
                tag, tys = min(vd.ctors, key=lambda c: len(c[1]))
            else:
                tag, tys = self.rng.choice(vd.ctors)
            return VariantVal(ty.name, tag, tuple(self.sample(t, depth + 1) for t in tys))
        if isinstance(ty, TFn):
            names = self.fn_names_by_type.get(ty)
            if not names:
                raise ValueError(f"no function value available for {ty}")
            return FnVal(self.rng.choice(names))
        raise ValueError(f"cannot sample value of type {ty}")

    def sample_inputs(self, param_types: list[Type]) -> list:
        return [self.sample(t) for t in param_types]
