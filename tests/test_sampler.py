from lucid.typechecker import typecheck
from loom.config import SamplerConfig
from loom.features import analyze
from loom.rng import Rng
from loom.sampler import Sampler


def test_feature_flags_respected():
    cfg = SamplerConfig(
        use_records=False, use_variants=False, use_match=False, use_cond=False,
        use_foreach=False, use_if=False, use_strings=False, use_higher_order=False,
        use_nested_lists=False, max_records=0, max_variants=0,
    )
    s = Sampler(cfg)
    for seed in range(300):
        m = s.sample_program(Rng(seed))
        typecheck(m)
        w = analyze(m.program)
        forbidden = {"record", "variant", "match", "cond", "foreach", "if",
                     "strings", "higher_order", "first_class_fn"}
        assert not (w.features & forbidden), f"seed {seed}: {w.features & forbidden}"


def test_strings_flag_off_means_no_str_literals():
    cfg = SamplerConfig(use_strings=False)
    s = Sampler(cfg)
    for seed in range(200):
        w = analyze(s.sample_program(Rng(seed)).program)
        assert "strings" not in w.features


def test_records_flag_isolated():
    cfg = SamplerConfig(use_variants=False, max_variants=0)
    s = Sampler(cfg)
    saw_record = False
    for seed in range(300):
        m = s.sample_program(Rng(seed))
        typecheck(m)
        w = analyze(m.program)
        assert "variant" not in w.features
        saw_record = saw_record or ("record" in w.features)
    assert saw_record  # records should still appear


def test_entry_has_no_function_typed_params():
    """IO sampling relies on entry params being plain value types."""
    from lucid.types import TFn
    s = Sampler()
    for seed in range(300):
        entry = s.sample_program(Rng(seed)).program.entry()
        for _, t in entry.params:
            assert not isinstance(t, TFn)
