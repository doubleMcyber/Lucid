import json
import os

from loom.evaluate import evaluate, extract_program
from loom.export import build_prompt, export_dataset, iter_records
from loom.pipeline import GenSpec, generate


def _make_dataset(tmp_path):
    out = str(tmp_path / "ds")
    generate(GenSpec(out_dir=out, n_examples=120, base_seed=1))
    return out


def test_export_split_is_leakage_free(tmp_path):
    ds = _make_dataset(tmp_path)
    exp = str(tmp_path / "exp")
    stats = export_dataset(ds, exp, tasks=["spec_to_code", "io_to_code"], test_pct=20)
    assert stats["train"] > 0 and stats["test"] > 0
    train = [json.loads(l) for l in open(stats["train_path"])]
    test = [json.loads(l) for l in open(stats["test_path"])]
    train_hashes = {e["ast_hash"] for e in train}
    test_hashes = {e["ast_hash"] for e in test}
    assert train_hashes.isdisjoint(test_hashes), "program leaked across split"
    # prompt/completion present and code tag in prompt
    for e in train[:5]:
        assert e["prompt"] and e["completion"]
        if e["task"] != "completion":
            assert e["completion"] == e["reference"]


def test_oracle_model_scores_perfect(tmp_path):
    """A model that emits the reference program must score 1.0 everywhere."""
    ds = _make_dataset(tmp_path)
    exp = str(tmp_path / "exp")
    export_dataset(ds, exp, tasks=["spec_to_code"], test_pct=30)
    test = [json.loads(l) for l in open(os.path.join(exp, "test.jsonl"))]
    items = [(rec, rec["reference"]) for rec in test]
    res = evaluate(items)
    assert res.parse_rate == 1.0
    assert res.typecheck_rate == 1.0
    assert res.exec_pass_at_1 == 1.0
    assert res.exact_match == 1.0


def test_garbage_model_scores_zero(tmp_path):
    ds = _make_dataset(tmp_path)
    exp = str(tmp_path / "exp")
    export_dataset(ds, exp, tasks=["spec_to_code"], test_pct=30)
    test = [json.loads(l) for l in open(os.path.join(exp, "test.jsonl"))]
    items = [(rec, "this is not lucid code at all !!!") for rec in test]
    res = evaluate(items)
    assert res.parse_rate == 0.0
    assert res.exec_pass_at_1 == 0.0


def test_extract_program_salvages_noisy_output():
    prog = "fn @f ($x : #Int) -> #Int = do\n  return add($x, 1) ;\nend @f ;"
    noisy = prog + "\n### Spec:\nblah blah\nmore noise"
    assert extract_program(noisy).strip() == prog
    # trailing partial line dropped
    truncated = prog + "\n  let $y : #Int = "
    from lucid.parser import parse
    parse(extract_program(truncated))  # must parse


def test_value_from_json_roundtrips():
    from lucid.interp import value_from_json, value_to_json
    from lucid.values import FnVal, RecordVal, VariantVal
    vals = [
        0, -5, 42, True, False, "hi", "", [1, 2, 3], [],
        [[1], [2, 3]], [True, False],
        RecordVal("P", (("x", 1), ("y", -2))),
        VariantVal("O", "Some", (7,)),
        VariantVal("O", "None", ()),
        FnVal("h0"),
        [RecordVal("P", (("x", 1),)), RecordVal("P", (("x", 2),))],
    ]
    for v in vals:
        assert value_from_json(value_to_json(v)) == v


def test_extract_program_keeps_long_valid_program():
    # regression: a long but valid program must not be truncated
    fns = "\n\n".join(
        f"fn @f{i} ($x : #Int) -> #Int = do\n  return add($x, {i}) ;\nend @f{i} ;"
        for i in range(40)
    )
    assert extract_program(fns) == fns


def test_empty_generation_not_credited():
    """Whitespace-only or stop-marker-only output is not a program: it must score
    0 on parse_rate and typecheck_rate (regression: an empty module parses and
    typechecks vacuously)."""
    rec = {"id": "x", "reference": "fn @main () -> #Int = do return 0 ; end @main ;",
           "eval_io": [{"input": [], "output": 0}]}
    for junk in ["", "   \n\n", "\n\n\n", "```", "<|im_end|>"]:
        res = evaluate([(rec, junk)])
        assert res.parse_rate == 0.0, junk
        assert res.typecheck_rate == 0.0, junk
        assert res.exec_pass_at_1 == 0.0, junk


def test_export_split_is_prompt_level_leakage_free(tmp_path):
    """No test prompt may also appear in train — distinct programs can collapse to
    identical prompts, which would hand the model a memorizable answer."""
    ds = _make_dataset(tmp_path)
    exp = str(tmp_path / "exp")
    export_dataset(ds, exp, tasks=["spec_to_code", "io_to_code"], test_pct=25)
    train = [json.loads(l) for l in open(os.path.join(exp, "train.jsonl"))]
    test = [json.loads(l) for l in open(os.path.join(exp, "test.jsonl"))]
    train_prompts = {e["prompt"] for e in train}
    test_prompts = {e["prompt"] for e in test}
    assert train_prompts.isdisjoint(test_prompts), "prompt leaked across split"


def test_io_to_code_eval_uses_held_out_io(tmp_path):
    """io_to_code test items carry held-out eval_io that is NOT shown in the
    prompt, so exec_pass@1 measures generalization, not reproduction."""
    from loom.export import PROMPT_K, _fmt_io
    ds = _make_dataset(tmp_path)
    exp = str(tmp_path / "exp")
    export_dataset(ds, exp, tasks=["io_to_code"], test_pct=40)
    test = [json.loads(l) for l in open(os.path.join(exp, "test.jsonl"))]
    assert test, "expected some io_to_code test items"
    for e in test:
        assert e["eval_io"], "io_to_code test item must have held-out IO"
        shown = _fmt_io(e["io_examples"], PROMPT_K)
        # every held-out input is beyond the first PROMPT_K shown in the prompt
        assert e["eval_io"] == e["io_examples"][PROMPT_K:]
        assert shown in e["prompt"]


def test_wrong_but_valid_program_fails_exec_not_parse(tmp_path):
    """A program that parses+typechecks but computes the wrong thing should pass
    parse/typecheck but fail exec_pass@1."""
    ds = _make_dataset(tmp_path)
    exp = str(tmp_path / "exp")
    export_dataset(ds, exp, tasks=["spec_to_code"], test_pct=40)
    test = [json.loads(l) for l in open(os.path.join(exp, "test.jsonl"))]
    # constant function returning a fixed int — valid but wrong for most specs
    wrong = "fn @main () -> #Int = do return 999999 ; end @main ;"
    items = [(rec, wrong) for rec in test]
    res = evaluate(items)
    assert res.parse_rate == 1.0
    assert res.typecheck_rate == 1.0
    assert res.exec_pass_at_1 < 1.0  # almost never matches
