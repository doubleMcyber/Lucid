"""The Lucid→Python transpiler must be behaviorally faithful: transpiled Python,
run on a program's IO inputs, reproduces the Lucid outputs exactly. Otherwise the
"matched baseline" is not actually matched."""

import json

from loom.config import SamplerConfig
from loom.export import iter_records
from loom.pipeline import GenSpec, generate
from loom.transpile_py import entry_name, py_signature, to_python
from lucid.parser import parse
from lucid.typechecker import typecheck


def _tiny() -> SamplerConfig:
    return SamplerConfig(
        use_bool=True, use_lists=True, use_records=False, use_variants=False,
        use_match=False, use_cond=True, use_foreach=True, use_if=True,
        use_strings=False, use_higher_order=False, use_nested_lists=False,
        min_helpers=0, max_helpers=0, min_params=1, max_params=2,
        max_block_stmts=2, max_loop_stmts=1, block_depth=1,
        expr_fuel=2, helper_fuel=1, max_list_lit=2, int_min=0, int_max=9,
    )


def _run_py(code: str, ent: str, inputs: list):
    g: dict = {}
    exec(code, g)
    got = g[ent](*[json.loads(json.dumps(x)) for x in inputs])
    return json.loads(json.dumps(got))


def test_transpiled_python_matches_lucid_io(tmp_path):
    out = str(tmp_path / "ds")
    generate(GenSpec(out_dir=out, n_examples=200, base_seed=11,
                     use_curriculum=False, sampler_config=_tiny(),
                     n_inputs=4, want_trace=False, want_repair=False,
                     want_refactor=False, want_completion=False))
    n = ok = skipped = 0
    for rec in iter_records(out):
        mod = parse(rec["program_canonical"])
        typecheck(mod)
        try:
            code, ent = to_python(mod), entry_name(mod)
        except NotImplementedError:
            skipped += 1
            continue
        for io in rec["io_examples"]:
            n += 1
            got = _run_py(code, ent, io["input"])
            assert json.dumps(got, sort_keys=True) == json.dumps(io["output"], sort_keys=True), \
                f"\n{rec['program_canonical']}\nin={io['input']} got={got} exp={io['output']}"
            ok += 1
    assert skipped == 0, "tiny config must transpile fully"
    assert n > 0 and ok == n


def test_total_semantics_div_mod_by_zero():
    src = ("fn @main ($a : #Int) -> #Int = do\n"
           "  return div($a, 0) ;\nend @main ;")
    mod = parse(src); typecheck(mod)
    assert _run_py(to_python(mod), entry_name(mod), [5]) == 0


def test_get_or_out_of_bounds_uses_default():
    src = ("fn @main ($xs : #List[#Int]) -> #Int = do\n"
           "  return get_or($xs, 99, 0) ;\nend @main ;")
    mod = parse(src); typecheck(mod)
    assert _run_py(to_python(mod), entry_name(mod), [[1, 2, 3]]) == 0


def test_py_signature_renders_python_types():
    src = ("fn @main ($xs : #List[#Int], $b : #Bool) -> #List[#Bool] = do\n"
           "  return list[#Bool]($b) ;\nend @main ;")
    mod = parse(src); typecheck(mod)
    assert py_signature(mod) == "(list[int], bool) -> list[bool]"
