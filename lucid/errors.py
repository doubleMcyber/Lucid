"""Lucid error hierarchy.

The boundaries matter for Loom: the validator distinguishes a *generator bug*
(a type error in a program the sampler claimed was well-typed — logged loudly)
from an *expected* runtime discard (resource bound hit during execution).
"""

from __future__ import annotations


class LucidError(Exception):
    """Base class for all Lucid toolchain errors."""


class LexError(LucidError):
    def __init__(self, msg: str, pos: int):
        super().__init__(f"lex error at {pos}: {msg}")
        self.pos = pos


class ParseError(LucidError):
    def __init__(self, msg: str, pos: int = -1):
        loc = f" at token {pos}" if pos >= 0 else ""
        super().__init__(f"parse error{loc}: {msg}")
        self.pos = pos


class TypeError_(LucidError):
    """Static type error. In Loom this signals a generator bug if it fires."""


class RuntimeError_(LucidError):
    """A runtime fault (e.g. resource bound exceeded). Programs hitting this
    are discarded by the validator; they are expected, not bugs."""


class ResourceError(RuntimeError_):
    """Execution exceeded a configured step/size/depth budget."""
