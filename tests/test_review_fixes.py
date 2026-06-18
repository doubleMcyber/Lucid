"""Regression tests for the 8 adversarial-review findings."""

import tempfile

from lucid.ast import (
    BoolLit, FnDecl, IntLit, Program, RecordDecl, RecordLit, Return,
)
from lucid.hashing import ast_hash
from lucid.parser import parse
from lucid.printer import print_program
from lucid.types import BOOL, INT, TRecord
from loom.astutil import rename_local_in_fn
from loom.config import SamplerConfig
from loom.curriculum import curriculum_stages
from loom.features import analyze, difficulty
from loom.pipeline import GenSpec, generate
from loom.rng import Rng
from loom.sampler import Sampler


def test_record_field_order_hash_invariant():
    # Finding 0: two record literals differing only in field order share a hash.
    A = Program((RecordDecl("R0", (("f0", INT), ("f1", BOOL))),
                 FnDecl("main", (), TRecord("R0"),
                        (Return(RecordLit("R0", (("f0", IntLit(1)), ("f1", BoolLit(True))))),))))
    B = Program((RecordDecl("R0", (("f0", INT), ("f1", BOOL))),
                 FnDecl("main", (), TRecord("R0"),
                        (Return(RecordLit("R0", (("f1", BoolLit(True)), ("f0", IntLit(1))))),))))
    assert ast_hash(A) == ast_hash(B)
    assert parse(print_program(A)).program == parse(print_program(B)).program


def test_nested_lists_gated():
    # Finding 1: use_nested_lists=False -> no nested-list types anywhere.
    s = Sampler(SamplerConfig(use_nested_lists=False))
    for seed in range(400):
        txt = print_program(s.sample_program(Rng(seed)).program)
        assert "#List[#List[" not in txt, f"nested list leaked at seed {seed}"


def test_rename_renames_match_binders():
    # Finding 2: rename must rewrite match-arm binder declarations too.
    src = ("variant #O = Some(#Int) | None() ; "
           "fn @main ($o : #O) -> #Int = do "
           "return match #O $o of case Some($v : #Int) -> $v case None() -> 0 end match ; "
           "end @main ;")
    m = parse(src)
    entry = m.program.entry()
    renamed = rename_local_in_fn(entry, "v", "r_v")
    out = print_program(Program(m.program.decls[:-1] + (renamed,)))
    # the renamed program must still parse + typecheck (binder + use both renamed)
    from lucid.typechecker import typecheck
    typecheck(parse(out))
    assert "$r_v" in out and "case Some($v " not in out


def test_stage0_is_pure_int_arithmetic():
    # Finding 3: stage 0 has no bool, lists, control flow, mutation.
    s = Sampler(curriculum_stages()[0])
    for seed in range(300):
        prog = s.sample_program(Rng(seed)).program
        w = analyze(prog)
        forbidden = {"strings", "foreach", "if", "cond", "record", "variant",
                     "match", "higher_order", "mutable_local"}
        assert not (w.features & forbidden), f"seed {seed}: {w.features & forbidden}"
        txt = print_program(prog)
        assert "#List[" not in txt and "#Bool" not in txt


def test_boost_does_not_force_higher_order_into_early_stages(tmp_path):
    # Finding 4: with coverage_guided on, early-stage records must not gain HOF.
    out = str(tmp_path / "ds")
    generate(GenSpec(out_dir=out, n_examples=120, base_seed=4))
    import json, os
    recs = []
    for name in os.listdir(out):
        if name.endswith(".jsonl"):
            recs += [json.loads(l) for l in open(os.path.join(out, name))]
    for r in recs:
        if r["curriculum_stage"] < 5:
            assert "higher_order" not in r["features"], r["id"]


def test_diversity_under_no_dedup_reflects_duplicates(tmp_path):
    # Finding 5: diversity_ratio is distinct-AST ratio over written records.
    cfg = SamplerConfig(use_bool=False, use_lists=False, use_records=False,
                        use_variants=False, use_match=False, use_cond=False,
                        use_foreach=False, use_if=False, use_strings=False,
                        use_higher_order=False, max_records=0, max_variants=0,
                        min_helpers=0, max_helpers=0, min_params=1, max_params=1,
                        max_block_stmts=0, expr_fuel=1, int_min=0, int_max=1)
    st = generate(GenSpec(out_dir=str(tmp_path / "d"), n_examples=40, base_seed=0,
                          use_curriculum=False, sampler_config=cfg, dedup=False,
                          want_trace=False, want_repair=False, want_refactor=False,
                          want_completion=False))
    assert st.coverage["diversity_ratio"] < 1.0
    assert st.coverage["duplicates_written"] > 0


def test_difficulty_spans_range_and_not_saturated():
    # Finding 6: difficulty exercises 1-10 and isn't ~all 10.
    s_easy = Sampler(curriculum_stages()[0])
    s_hard = Sampler(curriculum_stages()[5])
    easy = [difficulty(s_easy.sample_program(Rng(i)).program) for i in range(100)]
    hard = [difficulty(s_hard.sample_program(Rng(i)).program) for i in range(100)]
    assert max(easy) <= 4, "stage 0 should be easy"
    assert sum(easy) / len(easy) < sum(hard) / len(hard), "hard > easy on average"
    # across the mix, more than one bucket is used
    assert len(set(easy + hard)) >= 4
