from loom.coverage import CoverageTracker
from loom.curriculum import curriculum_stages
from loom.features import ALL_PRODUCTIONS, analyze
from loom.rng import Rng
from loom.sampler import Sampler
from lucid.hashing import ast_hash


def test_full_production_coverage_reached():
    s = Sampler()
    tracker = CoverageTracker()
    for seed in range(1500):
        prog = s.sample_program(Rng(seed)).program
        tracker.update(prog, ast_hash(prog))
    rep = tracker.report()
    assert rep["production_coverage"] >= 0.99, rep["missing_productions"]
    assert rep["type_interaction_count"] >= 5


def test_all_productions_nonempty_and_includes_builtins():
    assert len(ALL_PRODUCTIONS) > 50
    assert "builtin:add" in ALL_PRODUCTIONS
    assert "expr:match" in ALL_PRODUCTIONS
    assert "stmt:foreach" in ALL_PRODUCTIONS


def test_curriculum_is_monotonic_in_features():
    stages = curriculum_stages()
    flags = ["use_bool", "use_lists", "use_records", "use_variants", "use_match",
             "use_cond", "use_foreach", "use_if", "use_strings", "use_higher_order"]
    # later stages enable a superset of features of earlier ones
    enabled = [set(f for f in flags if getattr(st, f)) for st in stages]
    for i in range(1, len(enabled)):
        assert enabled[i - 1] <= enabled[i], f"stage {i} dropped a feature"


def test_curriculum_stage0_is_simple():
    s = Sampler(curriculum_stages()[0])
    for seed in range(200):
        w = analyze(s.sample_program(Rng(seed)).program)
        assert not (w.features & {"record", "variant", "match", "cond",
                                  "foreach", "if", "higher_order"})


def test_diversity_high_on_default_config():
    s = Sampler()
    tracker = CoverageTracker()
    for seed in range(500):
        prog = s.sample_program(Rng(seed)).program
        tracker.update(prog, ast_hash(prog))
    assert tracker.diversity_ratio() >= 0.95
