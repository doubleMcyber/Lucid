"""Loom command-line interface.

    python -m loom.cli generate --out data --n 1000 --seed 0
    python -m loom.cli inspect --seed 42
    python -m loom.cli verify --n 2000 --seed 0
    python -m loom.cli coverage --out data
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from lucid.hashing import ast_hash
from lucid.parser import parse
from lucid.printer import print_program
from lucid.typechecker import typecheck

from .config import SamplerConfig
from .features import ALL_PRODUCTIONS, analyze
from .pipeline import GenSpec, generate
from .rng import Rng
from .sampler import Sampler
from .validator import Validator


def cmd_generate(args) -> int:
    spec = GenSpec(
        out_dir=args.out,
        n_examples=args.n,
        base_seed=args.seed,
        shard_size=args.shard_size,
        use_curriculum=not args.flat,
        sampler_config=SamplerConfig() if args.flat else None,
        n_inputs=args.inputs,
        want_trace=not args.no_trace,
        want_completion=not args.no_completion,
        want_repair=not args.no_repair,
        want_refactor=not args.no_refactor,
        dedup=not args.no_dedup,
    )
    t0 = time.time()
    stats = generate(spec)
    dt = time.time() - t0
    print(f"wrote {stats.written} records to {args.out} in {dt:.1f}s "
          f"({stats.written / dt:.0f}/s)")
    print(f"  attempts={stats.attempts} duplicates={stats.duplicates} "
          f"discarded_no_io={stats.discarded_no_io} typecheck_bugs={stats.typecheck_bugs}")
    print(f"  production_coverage={stats.coverage['production_coverage']*100:.2f}% "
          f"diversity_ratio={stats.coverage['diversity_ratio']:.3f}")
    print(f"  manifest: {stats.manifest_path}")
    if stats.typecheck_bugs:
        print("  WARNING: typecheck bugs detected (generator invariant violated)!",
              file=sys.stderr)
        return 1
    return 0


def cmd_inspect(args) -> int:
    cfg = SamplerConfig()
    m = Sampler(cfg).sample_program(Rng(args.seed))
    canon = print_program(m.program)
    print(canon)
    print("\n--- analysis ---")
    typecheck(m)
    w = analyze(m.program)
    print("typechecks: yes")
    print("features:", sorted(w.features))
    print("difficulty:", __import__("loom.features", fromlist=["difficulty"]).difficulty(m.program))
    rep = Validator(n_inputs=3).validate(m, Rng(args.seed + 1))
    print("accepted:", rep.accepted, "io_pairs:", len(rep.io))
    for p in rep.io[:3]:
        print("  io:", json.dumps({"input": p.input_json, "output": p.output_json}))
    return 0


def cmd_verify(args) -> int:
    """Self-check: validity (typecheck) + bijectivity over N programs."""
    s = Sampler()
    tc_fail = rt_fail = 0
    t0 = time.time()
    for seed in range(args.n):
        m = s.sample_program(Rng(seed))
        try:
            typecheck(m)
        except Exception as ex:
            tc_fail += 1
            if tc_fail <= 3:
                print("TYPECHECK FAIL", seed, ex, file=sys.stderr)
            continue
        canon = print_program(m.program)
        m2 = parse(canon)
        if m2.program != m.program or print_program(m2.program) != canon:
            rt_fail += 1
    dt = time.time() - t0
    print(f"verified {args.n} programs in {dt:.1f}s")
    print(f"  typecheck validity: {100*(args.n-tc_fail)/args.n:.3f}%  (failures={tc_fail})")
    print(f"  bijectivity:        {100*(args.n-rt_fail)/args.n:.3f}%  (failures={rt_fail})")
    return 0 if (tc_fail == 0 and rt_fail == 0) else 1


def cmd_coverage(args) -> int:
    with open(f"{args.out}/manifest.json") as f:
        man = json.load(f)
    cov = man.get("coverage", {})
    print(json.dumps(cov, indent=2))
    print(f"\nproductions universe size: {len(ALL_PRODUCTIONS)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loom", description="Lucid corpus generator")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a dataset")
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--shard-size", type=int, default=1000)
    g.add_argument("--inputs", type=int, default=4)
    g.add_argument("--flat", action="store_true", help="single config (no curriculum)")
    g.add_argument("--no-trace", action="store_true")
    g.add_argument("--no-completion", action="store_true")
    g.add_argument("--no-repair", action="store_true")
    g.add_argument("--no-refactor", action="store_true")
    g.add_argument("--no-dedup", action="store_true")
    g.set_defaults(func=cmd_generate)

    i = sub.add_parser("inspect", help="print one sample program")
    i.add_argument("--seed", type=int, default=0)
    i.set_defaults(func=cmd_inspect)

    v = sub.add_parser("verify", help="check validity + bijectivity")
    v.add_argument("--n", type=int, default=1000)
    v.add_argument("--seed", type=int, default=0)
    v.set_defaults(func=cmd_verify)

    c = sub.add_parser("coverage", help="print coverage from a dataset manifest")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_coverage)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
