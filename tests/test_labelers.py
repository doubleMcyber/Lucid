from lucid.parser import parse
from lucid.printer import print_program
from lucid.typechecker import typecheck
from loom.labelers import (
    completion_pairs, execution_trace, io_examples, refactor_pairs,
    repair_pairs, spec_for, validated_paraphrase,
)
from loom.rng import Rng
from loom.sampler import Sampler
from loom.validator import Validator


def _accepted(seed_start=0):
    s = Sampler()
    v = Validator(n_inputs=4)
    for seed in range(seed_start, seed_start + 200):
        rng = Rng(seed)
        m = s.sample_program(rng.spawn(1))
        rep = v.validate(m, rng.spawn(2))
        if rep.accepted:
            return m, rep, rng
    raise AssertionError("no accepted program found")


def test_io_examples_present():
    m, rep, _ = _accepted()
    io = io_examples(rep)
    assert io and all("input" in p and "output" in p for p in io)


def test_completion_split_reconstructs():
    m, rep, rng = _accepted()
    canon = print_program(m.program)
    pairs = completion_pairs(canon, rng.spawn(3))
    for p in pairs:
        if p["kind"] == "completion":
            assert p["prefix"] + p["completion"] == canon
        if p["kind"] == "infill":
            assert p["prefix"] + p["middle"] + p["suffix"] == canon


def test_repair_breaks_are_verified():
    # search for a program that yields both repair kinds
    s = Sampler(); v = Validator(n_inputs=4)
    found_type = found_behavior = False
    for seed in range(400):
        rng = Rng(seed)
        m = s.sample_program(rng.spawn(1))
        rep = v.validate(m, rng.spawn(2))
        if not rep.accepted:
            continue
        canon = print_program(m.program)
        for r in repair_pairs(m, canon, rep, rng.spawn(4)):
            assert r["fixed"] == canon
            if r["kind"] == "repair_type":
                found_type = True
                # broken version must actually fail to typecheck
                try:
                    typecheck(parse(r["broken"]))
                    raise AssertionError("repair_type broken still typechecks")
                except Exception:
                    pass
            if r["kind"] == "repair_behavior":
                found_behavior = True
                assert r["broken_output"] != r["fixed_output"]
        if found_type and found_behavior:
            break
    assert found_type and found_behavior


def test_refactor_is_behaviorally_equivalent():
    s = Sampler(); v = Validator(n_inputs=4)
    checked = 0
    for seed in range(200):
        rng = Rng(seed)
        m = s.sample_program(rng.spawn(1))
        rep = v.validate(m, rng.spawn(2))
        if not rep.accepted:
            continue
        canon = print_program(m.program)
        for r in refactor_pairs(m, canon, rep, rng.spawn(5)):
            assert r["relation"] == "equivalent"
            assert r["a"] == canon
            # parsing b must succeed and typecheck
            typecheck(parse(r["b"]))
            checked += 1
        if checked > 20:
            break
    assert checked > 0


def test_spec_signature_matches_entry():
    m, rep, _ = _accepted()
    spec = spec_for(m)
    entry = m.program.entry()
    expected = "({}) -> {}".format(
        ", ".join(str(t) for _, t in entry.params), str(entry.ret))
    assert spec["type_signature"] == expected


def test_back_translation_keeps_only_verified():
    m, rep, _ = _accepted()
    canon = print_program(m.program)
    spec = spec_for(m)
    good = validated_paraphrase(spec, m, rep, lambda s: "p", lambda p: canon)
    assert good["spec_verified"] is True
    # a synthesizer that returns nonsense -> not verified
    bad = validated_paraphrase(spec, m, rep, lambda s: "p", lambda p: "not lucid code {{{")
    assert bad["spec_verified"] is False


def test_trace_labeler():
    m, rep, _ = _accepted()
    tr = execution_trace(m, rep)
    assert tr is not None
    assert tr["trace"] and "input" in tr and "output" in tr
