"""Canonical AST hashing.

Because the surface form is bijective with the AST, the canonical text is a
faithful fingerprint of the program. The AST hash is `sha256(canonical_text)`,
used for dedup (PRD §8.4) and provenance (PRD §8.5). Two programs share a hash
iff they have the identical canonical form iff they are the identical AST.
"""

from __future__ import annotations

import hashlib

from .ast import Program
from .printer import print_program


def ast_hash(program: Program) -> str:
    canon = print_program(program)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def text_hash(canonical_text: str) -> str:
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
