"""Every hand-authored example parses, typechecks, and executes on sampled
inputs without a runtime type error (M2 deliverable: examples typecheck+execute)."""

import glob
import os

import pytest

from lucid.errors import ResourceError
from lucid.interp import Interpreter
from lucid.parser import parse
from lucid.typechecker import typecheck
from lucid.types import TFn
from loom.rng import Rng
from loom.value_sampler import ValueSampler

EX_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
EXAMPLES = sorted(glob.glob(os.path.join(EX_DIR, "*.lucid")))


@pytest.mark.parametrize("path", EXAMPLES)
def test_example_executes(path):
    with open(path) as f:
        m = parse(f.read())
    typecheck(m)
    entry = m.program.entry()
    ptypes = [t for _, t in entry.params]
    fn_by_type = {}
    for fn in m.program.functions():
        ft = TFn(tuple(t for _, t in fn.params), fn.ret)
        fn_by_type.setdefault(ft, []).append(fn.name)
    vs = ValueSampler(m.tenv, Rng(hash(path) & 0xFFFF), fn_by_type)
    ran = 0
    for _ in range(8):
        inputs = vs.sample_inputs(ptypes)
        try:
            Interpreter(m).run_entry(inputs)
            ran += 1
        except ResourceError:
            pass
    assert ran > 0, f"{os.path.basename(path)} never executed within bounds"
