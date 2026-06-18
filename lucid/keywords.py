"""Reserved bare words. The lexer emits every bare word as a NAME token; the
parser uses this set to tell structural/expression keywords apart from built-in
calls and variant tags. Keeping the set here (not in the lexer) preserves the
"lexical class is context-free" property: a NAME is a NAME regardless of role.
"""

from __future__ import annotations

KEYWORDS: frozenset[str] = frozenset(
    {
        # structural
        "fn", "do", "end", "let", "var", "set", "foreach", "in",
        "if", "else", "return", "record", "variant",
        # expression forms
        "true", "false", "cond", "then", "match", "of", "case",
        "list", "new", "tag", "get",
    }
)


def is_keyword(name: str) -> bool:
    return name in KEYWORDS
