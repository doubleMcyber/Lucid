"""Bijectivity property tests (PRD §7.2, M1 deliverable).

  parse(print(ast)) == ast                  (round-trip through text)
  print(parse(canonical)) == canonical      (canonical text is a fixed point)
  print is idempotent

Tested on the hand-authored examples and on a large sample of generated ASTs.
"""

import glob
import os

import pytest

from lucid.parser import parse
from lucid.printer import fence_string, print_program
from loom.rng import Rng
from loom.sampler import Sampler

EX_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
EXAMPLES = sorted(glob.glob(os.path.join(EX_DIR, "*.lucid")))


@pytest.mark.parametrize("path", EXAMPLES)
def test_examples_roundtrip(path):
    with open(path) as f:
        src = f.read()
    m = parse(src)
    canon = print_program(m.program)
    m2 = parse(canon)
    assert m2.program == m.program          # parse(print(ast)) == ast
    assert print_program(m2.program) == canon  # canonical fixpoint
    assert print_program(parse(canon).program) == canon  # idempotent


def test_generated_bijectivity():
    s = Sampler()
    for seed in range(400):
        m = s.sample_program(Rng(seed))
        canon = print_program(m.program)
        m2 = parse(canon)
        assert m2.program == m.program, f"AST round-trip failed at seed {seed}"
        assert print_program(m2.program) == canon, f"fixpoint failed at seed {seed}"


def test_fence_string_minimal_and_escape_free():
    assert fence_string("plain") == '"plain"'
    # content with a quote escalates to one hash fence
    assert fence_string('a"b') == '#"a"b"#'
    # content with quote+hash escalates further
    assert fence_string('a"#b') == '##"a"#b"##'
    # round-trips through the lexer/parser inside a program
    from lucid import lexer as lx
    for s in ["", "x", 'a"b', 'q"#"##end', "multi\nline"]:
        tok = lx.tokenize(fence_string(s))[0]
        assert tok.kind == lx.STR and tok.value == s
