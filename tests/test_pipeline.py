import json
import os

from lucid.parser import parse
from lucid.typechecker import typecheck
from loom.pipeline import GenSpec, generate


def _load_records(out_dir):
    recs = []
    for name in sorted(os.listdir(out_dir)):
        if name.endswith(".jsonl"):
            with open(os.path.join(out_dir, name)) as f:
                recs.extend(json.loads(line) for line in f if line.strip())
    return recs


def test_end_to_end_schema_and_validity(tmp_path):
    out = str(tmp_path / "ds")
    stats = generate(GenSpec(out_dir=out, n_examples=120, base_seed=3, shard_size=50))
    assert stats.typecheck_bugs == 0
    assert stats.written == 120
    recs = _load_records(out)
    assert len(recs) == 120
    required = {
        "id", "language_version", "generator_version", "seed", "ast_hash",
        "program_canonical", "type_signature", "io_examples", "spec_structured",
        "features", "difficulty", "curriculum_stage", "trace_available",
        "spec_verified", "pair_types",
    }
    for r in recs:
        assert required <= set(r), f"missing {required - set(r)}"
        # every recorded program is itself valid
        m = parse(r["program_canonical"])
        typecheck(m)
        assert r["io_examples"], "must have >=1 IO pair"
        assert "spec_to_code" in r["pair_types"]
        assert "io_to_code" in r["pair_types"]


def test_reproducible_bytes(tmp_path):
    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    generate(GenSpec(out_dir=a, n_examples=80, base_seed=11, shard_size=1000))
    generate(GenSpec(out_dir=b, n_examples=80, base_seed=11, shard_size=1000))
    for name in os.listdir(a):
        if name.endswith(".jsonl"):
            with open(os.path.join(a, name), "rb") as f1, open(os.path.join(b, name), "rb") as f2:
                assert f1.read() == f2.read(), f"shard {name} differs across runs"


def test_dedup_no_duplicate_hashes(tmp_path):
    out = str(tmp_path / "ds")
    generate(GenSpec(out_dir=out, n_examples=150, base_seed=5))
    recs = _load_records(out)
    hashes = [r["ast_hash"] for r in recs]
    assert len(hashes) == len(set(hashes)), "duplicate ast_hash in dataset"


def test_manifest_coverage(tmp_path):
    out = str(tmp_path / "ds")
    generate(GenSpec(out_dir=out, n_examples=300, base_seed=0))
    with open(os.path.join(out, "manifest.json")) as f:
        man = json.load(f)
    cov = man["coverage"]
    assert cov["production_coverage"] >= 0.99
    assert cov["diversity_ratio"] >= 0.95
    assert man["typecheck_bugs"] == 0


def test_curriculum_orders_easy_to_hard(tmp_path):
    out = str(tmp_path / "ds")
    generate(GenSpec(out_dir=out, n_examples=180, base_seed=2))
    recs = _load_records(out)
    stages = [r["curriculum_stage"] for r in recs]
    # non-decreasing curriculum stage across the emitted stream
    assert stages == sorted(stages)
    # stage 0 programs use no advanced features
    for r in recs:
        if r["curriculum_stage"] == 0:
            adv = {"record", "variant", "match", "cond", "higher_order",
                   "first_class_fn", "foreach", "if"}
            assert not (set(r["features"]) & adv)
