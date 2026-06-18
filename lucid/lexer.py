"""Lucid lexer.

Design constraint (PRD §7.1): *no context-dependent lexing*. A token's lexical
class is decided by its own leading character(s), never by what came before:

    $x      -> LOCAL    (sigil $)
    @f      -> FUNC     (sigil @)
    #Int    -> TYPE     (sigil # followed by a letter)
    %field  -> FIELD    (sigil %)
    "..."   -> STR      (quote, or # run leading into a quote: raw fenced form)
    123     -> INT      (digit)
    add     -> NAME     (bare word: keyword/builtin/tag, disambiguated at parse)
    -> ( ) [ ] { } , ; = : |   -> punctuation

Whitespace (PRD §7.1) is purely a separator and carries no meaning.

String literals are escape-free (PRD §7.1): a raw, self-delimiting fenced form
modeled on Rust raw strings. The literal opens with k>=0 `#` then `"`, and
closes with `"` then the same k `#`. Canonical form uses the minimal k such that
the content cannot contain the closing delimiter — so there is never any in-band
escaping for the model to get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import LexError

# Token kinds.
LOCAL = "LOCAL"
FUNC = "FUNC"
TYPE = "TYPE"
FIELD = "FIELD"
NAME = "NAME"
INT = "INT"
STR = "STR"
LPAREN = "LPAREN"
RPAREN = "RPAREN"
LBRACK = "LBRACK"
RBRACK = "RBRACK"
LBRACE = "LBRACE"
RBRACE = "RBRACE"
COMMA = "COMMA"
SEMI = "SEMI"
EQ = "EQ"
ARROW = "ARROW"
COLON = "COLON"
PIPE = "PIPE"
EOF = "EOF"

_PUNCT_SINGLE = {
    "(": LPAREN,
    ")": RPAREN,
    "[": LBRACK,
    "]": RBRACK,
    "{": LBRACE,
    "}": RBRACE,
    ",": COMMA,
    ";": SEMI,
    "=": EQ,
    ":": COLON,
    "|": PIPE,
}

_WS = set(" \t\r\n\f\v")


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    pos: int

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Token({self.kind}, {self.value!r}, @{self.pos})"


def _is_ident_start(c: str) -> bool:
    return c.isalpha() or c == "_"


def _is_ident_cont(c: str) -> bool:
    return c.isalnum() or c == "_"


def tokenize(src: str) -> list[Token]:
    toks: list[Token] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c in _WS:
            i += 1
            continue

        # Sigiled identifiers.
        if c in "$@%":
            kind = {"$": LOCAL, "@": FUNC, "%": FIELD}[c]
            j = i + 1
            if j >= n or not _is_ident_start(src[j]):
                raise LexError(f"expected identifier after '{c}'", i)
            j += 1
            while j < n and _is_ident_cont(src[j]):
                j += 1
            toks.append(Token(kind, src[i + 1 : j], i))
            i = j
            continue

        # '#': either a type sigil (#Name) or a raw string ( #"..."# ).
        if c == "#":
            nxt = src[i + 1] if i + 1 < n else ""
            if nxt == "#" or nxt == '"':
                tok, i = _read_string(src, i)
                toks.append(tok)
                continue
            if _is_ident_start(nxt):
                j = i + 2
                while j < n and _is_ident_cont(src[j]):
                    j += 1
                toks.append(Token(TYPE, src[i + 1 : j], i))
                i = j
                continue
            raise LexError("expected type name or string after '#'", i)

        # Bare quote string.
        if c == '"':
            tok, i = _read_string(src, i)
            toks.append(tok)
            continue

        # Integer literal (non-negative; negatives use neg(...)).
        if c.isdigit():
            j = i + 1
            while j < n and src[j].isdigit():
                j += 1
            # Reject things like 12abc — identifiers cannot start with a digit,
            # so a digit run touching a letter is a malformed token.
            if j < n and _is_ident_cont(src[j]):
                raise LexError("malformed number", i)
            toks.append(Token(INT, src[i:j], i))
            i = j
            continue

        # Bare word: keyword / builtin / variant tag (resolved by the parser).
        if _is_ident_start(c):
            j = i + 1
            while j < n and _is_ident_cont(src[j]):
                j += 1
            toks.append(Token(NAME, src[i:j], i))
            i = j
            continue

        # Arrow.
        if c == "-":
            if i + 1 < n and src[i + 1] == ">":
                toks.append(Token(ARROW, "->", i))
                i += 2
                continue
            raise LexError("'-' is not an operator; use neg(...) or '->'", i)

        # Single-char punctuation.
        if c in _PUNCT_SINGLE:
            toks.append(Token(_PUNCT_SINGLE[c], c, i))
            i += 1
            continue

        raise LexError(f"unexpected character {c!r}", i)

    toks.append(Token(EOF, "", n))
    return toks


def _read_string(src: str, i: int) -> tuple[Token, int]:
    """Read a raw fenced string starting at index i. Returns (token, new_index)."""
    start = i
    n = len(src)
    k = 0
    while i < n and src[i] == "#":
        k += 1
        i += 1
    if i >= n or src[i] != '"':
        raise LexError("malformed string fence", start)
    i += 1  # consume opening quote
    closer = '"' + "#" * k
    end = src.find(closer, i)
    if end == -1:
        raise LexError("unterminated string literal", start)
    content = src[i:end]
    return Token(STR, content, start), end + len(closer)
