"""The matched Python baseline: faithful oracle, working sandbox, leak-free split."""

import json
import os

from loom.config import SamplerConfig
from loom.evaluate_py import evaluate_py, extract_program_py
from loom.export_py import export_python_dataset
from loom.leakcheck import audit_split
from loom.pipeline import GenSpec, generate
from loom.sandbox import run_one


def _tiny() -> SamplerConfig:
    return SamplerConfig(
        use_bool=True, use_lists=True, use_records=False, use_variants=False,
        use_match=False, use_cond=True, use_foreach=True, use_if=True,
        use_strings=False, use_higher_order=False, use_nested_lists=False,
        min_helpers=0, max_helpers=0, min_params=1, max_params=2,
        max_block_stmts=2, max_loop_stmts=1, block_depth=1,
        expr_fuel=2, helper_fuel=1, max_list_lit=2, int_min=0, int_max=9,
    )


def _make(tmp_path):
    raw = str(tmp_path / "raw")
    generate(GenSpec(out_dir=raw, n_examples=160, base_seed=5,
                     use_curriculum=False, sampler_config=_tiny(),
                     n_inputs=4, want_trace=False, want_repair=False,
                     want_refactor=False, want_completion=False))
    exp = str(tmp_path / "exp")
    stats = export_python_dataset(raw, exp, tasks=["spec_to_code", "io_to_code"], test_pct=35)
    return exp, stats


def test_python_oracle_scores_perfect(tmp_path):
    exp, _ = _make(tmp_path)
    test = [json.loads(l) for l in open(os.path.join(exp, "test.jsonl"))]
    assert test
    items = [(rec, rec["reference"]) for rec in test]
    res = evaluate_py(items)
    assert res.parse_rate == 1.0
    assert res.runs_rate == 1.0
    assert res.exec_pass_at_1 == 1.0, res.summary()


def test_python_garbage_scores_zero(tmp_path):
    exp, _ = _make(tmp_path)
    test = [json.loads(l) for l in open(os.path.join(exp, "test.jsonl"))]
    res = evaluate_py([(rec, "this is not python !!!") for rec in test])
    assert res.parse_rate == 0.0
    assert res.exec_pass_at_1 == 0.0


def test_python_empty_not_credited(tmp_path):
    exp, _ = _make(tmp_path)
    test = [json.loads(l) for l in open(os.path.join(exp, "test.jsonl"))]
    rec = test[0]
    for junk in ["", "   \n\n", "```", "# just a comment\n"]:
        res = evaluate_py([(rec, junk)])
        assert res.parse_rate == 0.0, junk


def test_python_export_split_leak_free(tmp_path):
    exp, _ = _make(tmp_path)
    rep = audit_split(os.path.join(exp, "train.jsonl"), os.path.join(exp, "test.jsonl"))
    assert rep.clean, rep.summary()
    assert rep.n_test > 0


def test_sandbox_blocks_infinite_loop():
    code = "def main(x):\n    while True:\n        pass\n    return x"
    out = run_one(code, "main", [[1]], timeout=2.0)
    assert out["status"] in ("timeout", "error")


def test_sandbox_blocks_open():
    code = "def main(x):\n    return open('/etc/passwd').read()"
    out = run_one(code, "main", [[1]])
    # restricted builtins => open is undefined => per-input error, not a crash
    assert out["status"] == "ok"
    assert out["outputs"][0]["ok"] is False


def test_sandbox_runs_normal_code():
    code = "def main(xs):\n    return sum(xs)"
    # one input case whose single argument is the list [1, 2, 3]
    out = run_one(code, "main", [[[1, 2, 3]]])
    assert out["status"] == "ok"
    assert out["outputs"][0] == {"ok": True, "output": 6}


def test_extract_python_salvages_noise():
    prog = "def main(x):\n    return x + 1"
    noisy = prog + "\n### Spec:\nblah"
    assert extract_program_py(noisy).strip() == prog
