"""Core language property tests (PRD §4 H1 well-formedness, §7.5 totality;
M2 deliverable: well-typed -> no runtime type error, guaranteed termination)."""

from lucid.errors import ResourceError, RuntimeError_
from lucid.interp import Interpreter
from lucid.parser import parse
from lucid.printer import print_program
from lucid.typechecker import typecheck
from lucid.types import TFn
from loom.rng import Rng
from loom.sampler import Sampler
from loom.value_sampler import ValueSampler


def test_validity_by_construction():
    """Every generated program type-checks (H1 / generator validity ~100%)."""
    s = Sampler()
    fails = 0
    for seed in range(1500):
        m = s.sample_program(Rng(seed))
        try:
            typecheck(m)
        except Exception:
            fails += 1
    assert fails == 0, f"{fails}/1500 generated programs failed to typecheck"


def test_welltyped_implies_no_runtime_type_error_and_terminates():
    """Run typechecked programs on sampled inputs. The ONLY acceptable failure
    is ResourceError (the totality safety net). Any other exception would be a
    type-safety hole. Completion of the loop also witnesses termination."""
    s = Sampler()
    runs = 0
    for seed in range(800):
        m = s.sample_program(Rng(seed))
        typecheck(m)
        entry = m.program.entry()
        ptypes = [t for _, t in entry.params]
        fn_by_type = {}
        for fn in m.program.functions():
            ft = TFn(tuple(t for _, t in fn.params), fn.ret)
            fn_by_type.setdefault(ft, []).append(fn.name)
        vs = ValueSampler(m.tenv, Rng(seed).spawn(9), fn_by_type)
        for k in range(3):
            inputs = vs.sample_inputs(ptypes)
            try:
                Interpreter(m).run_entry(inputs)
                runs += 1
            except ResourceError:
                pass  # allowed: bounded execution tripped a budget
            except RuntimeError_ as ex:
                # RuntimeError_ that is not a ResourceError should not happen
                raise AssertionError(f"runtime fault at seed {seed}: {ex}")
    assert runs > 0


def test_sampler_determinism():
    """Same seed -> identical program (canonical bytes)."""
    s = Sampler()
    for seed in [0, 1, 2, 99, 1000]:
        a = print_program(s.sample_program(Rng(seed)).program)
        b = print_program(s.sample_program(Rng(seed)).program)
        assert a == b


def test_no_recursion_in_generated_programs():
    """The DAG/no-recursion rule (totality) holds: every call/fnref targets an
    earlier-declared function."""
    from lucid.ast import Call, FnRef
    from loom.astutil import transform_program_exprs
    s = Sampler()
    for seed in range(300):
        prog = s.sample_program(Rng(seed)).program
        fns = prog.functions()
        order = {f.name: i for i, f in enumerate(fns)}
        for i, fn in enumerate(fns):
            refs = []

            def collect(e):
                if isinstance(e, (Call, FnRef)):
                    refs.append(e.name)
                return e
            from lucid.ast import Program
            transform_program_exprs(Program((fn,)), collect)
            for name in refs:
                assert order[name] < i, f"@{fn.name} references later/own @{name}"
