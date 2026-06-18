"""Tokenizer-friendliness analysis (PRD §7.1 open question).

Measures how a real BPE tokenizer (Qwen) segments canonical Lucid programs:
tokens per program and tokens per character, and how often Lucid's reserved
keywords/builtins are single tokens. This is the empirical lever on the
"single-token keywords" design goal and on whether Lucid is compact under a
tokenizer it was not co-designed with.

Run: SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())') \
     python scripts/tokenizer_analysis.py
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer  # noqa: E402

from lucid.builtins_def import BUILTINS  # noqa: E402
from lucid.keywords import KEYWORDS  # noqa: E402
from lucid.printer import print_program  # noqa: E402
from loom.rng import Rng  # noqa: E402
from loom.sampler import Sampler  # noqa: E402

MODEL = os.environ.get("LUCID_MODEL", "Qwen/Qwen3-0.6B-Base")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    s = Sampler()
    progs = [print_program(s.sample_program(Rng(i)).program) for i in range(300)]

    tok_per_prog, tok_per_char = [], []
    for p in progs:
        ids = tok.encode(p)
        tok_per_prog.append(len(ids))
        tok_per_char.append(len(ids) / max(1, len(p)))

    print(f"model: {MODEL}")
    print(f"programs: {len(progs)}")
    print(f"tokens/program: mean={statistics.mean(tok_per_prog):.1f} "
          f"median={statistics.median(tok_per_prog):.0f} "
          f"max={max(tok_per_prog)}")
    print(f"tokens/char:    mean={statistics.mean(tok_per_char):.3f} "
          f"(lower is more compact)")

    # How many reserved words are single tokens? (with and without a leading space)
    def single(word: str) -> bool:
        return len(tok.encode(word)) == 1 or len(tok.encode(" " + word)) == 1

    kw_single = sum(single(k) for k in KEYWORDS)
    bi_single = sum(single(b) for b in BUILTINS)
    print(f"keywords single-token: {kw_single}/{len(KEYWORDS)}")
    print(f"builtins single-token: {bi_single}/{len(BUILTINS)}")

    # Sigil cost: do $x/@f/#T/%f tokenize cheaply?
    for sig, name in [("$x", "local"), ("@f", "func"), ("#Int", "type"), ("%f", "field")]:
        print(f"  sigil {name:6} {sig!r}: {len(tok.encode(sig))} tokens")


if __name__ == "__main__":
    main()
