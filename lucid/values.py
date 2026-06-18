"""Runtime value representations for the Lucid interpreter.

Mapping from Lucid types to Python values:

    #Int        -> int
    #Bool       -> bool
    #Str        -> str
    #List[#T]   -> list
    #Rec        -> RecordVal (ordered, hashable)
    #Var        -> VariantVal (hashable)
    #Fn[...]    -> FnVal (a reference to a declared user function)

Records and variants are frozen/hashable so values can be compared and put in
IO-dedup sets. Lists are plain Python lists (compared with ==).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecordVal:
    type_name: str
    fields: tuple[tuple[str, Any], ...]  # ordered (name, value)

    def get(self, name: str) -> Any:
        for n, v in self.fields:
            if n == name:
                return v
        raise KeyError(name)


@dataclass(frozen=True)
class VariantVal:
    type_name: str
    tag: str
    args: tuple[Any, ...]


@dataclass(frozen=True)
class FnVal:
    """A first-class function value: a reference to a declared user function."""

    name: str
