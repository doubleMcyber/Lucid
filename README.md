# Lucid + Loom

**Lucid** is a model-native programming language — designed so a sequence model
can read and write it *correctly*, token by token, with minimal long-range
bookkeeping and zero local ambiguity. **Loom** is a type-directed corpus
generator that emits unlimited, fully-labeled, executable, diverse Lucid programs
for training and evaluating code models.

This repository implements the PRD in `ai_native_language_prd.md`.

## Layout

```
lucid/      the toolchain — the single source of truth for the language
  lexer.py parser.py printer.py     front end (context-free lex, bijective print)
  types.py ast.py typechecker.py    static, strong, total type system
  interp.py builtins_def.py values.py   total interpreter + step tracer
  hashing.py errors.py keywords.py

loom/       the corpus generator (consumes the toolchain as a library)
  sampler.py value_sampler.py       type-directed AST + value sampling
  validator.py                      typecheck + execute under resource bounds
  labelers.py                       IO, trace, completion, repair, refactor, spec
  coverage.py curriculum.py         coverage/diversity + easy→hard schedule
  writer.py pipeline.py cli.py      sharded JSONL, provenance, orchestration
  astutil.py features.py config.py rng.py

examples/   ~20 hand-authored canonical programs
tests/      129 tests (property, unit, end-to-end)
SPEC.md     language spec + grammar      DATASET_SCHEMA.md   record schema
```

## Quickstart

```bash
# generate a curriculum'd dataset (sharded JSONL + manifest)
python -m loom.cli generate --out data --n 1000 --seed 0

# print + analyze one sampled program
python -m loom.cli inspect --seed 42

# self-check: validity (typecheck) + bijectivity over N programs
python -m loom.cli verify --n 2000 --seed 0

# coverage report from a generated dataset
python -m loom.cli coverage --out data

# run the test suite
python -m pytest
```

## Using the toolchain directly

```python
from lucid.parser import parse
from lucid.typechecker import typecheck
from lucid.interp import run
from lucid.printer import print_program

m = parse(open("examples/sum_list.lucid").read())
typecheck(m)
print(run(m, [[1, 2, 3, 4]]).value)   # -> 10
```

## PRD success metrics (verified by the test suite)

| Metric (PRD §4) | Target | Achieved |
|---|---|---|
| Validity rate (typecheck) | ≥ 99.9% | **100%** by construction (3000+ programs) |
| Bijectivity `parse∘print = id` | — | **100%** |
| Well-typed ⇒ no runtime type error | — | holds (only `ResourceError` allowed) |
| Throughput (simple programs) | ≥ 1,000/s/core | **~1,100/s** IO-labeled; ~580/s with all 6 pair types; ~120/s full curriculum (measured, single core, this machine) |
| Diversity (distinct-AST ratio over written shard) | ≥ 0.95 | **~1.0** (dedup on) |
| Grammar-production coverage | ≥ 99% | **100%** (69/69) |
| Reproducibility (identical bytes) | — | **byte-identical** across `PYTHONHASHSEED` |
| Spec pairs per program | ≥ 1 verified | **1** templated (+ optional validated paraphrase) |

## Learnability experiment (PRD M6 / H1 / H2)

Does a small model actually *learn* Lucid from Loom data? A LoRA fine-tune of
**Qwen3-0.6B** on Apple-Silicon **MPS** (~40 min), evaluated by the toolchain on
held-out tasks (`scripts/experiment.py`, full write-up in `RESULTS.md`):

| metric | base model | fine-tuned |
|---|---|---|
| parse-rate (any valid Lucid) | **0%** | **100%** |
| typecheck-rate (H1) | **0%** | **80–98%** |
| executed pass@1 (H2) | **0%** | **25–43%** |

The base model never produces valid Lucid; after fine-tuning it **always** emits
parseable code, usually type-correct, and — from IO examples — code that runs and
reproduces a **held-out** input nearly half the time. The eval harness was
adversarially audited and hardened (leakage-free at the prompt level, held-out IO,
empty output rejected); see `RESULTS.md` for setup, the integrity notes, caveats,
and MPS engineering notes; `IMPROVEMENTS.md` for the gap analysis and roadmap
(incl. the matched-baseline pipeline for H3/H4).

```bash
# tokenizer-friendliness analysis (PRD §7.1) and the experiment
python scripts/tokenizer_analysis.py
python scripts/experiment.py run --n 6000   # prepare → base eval → train → tuned eval
```

## Notes

- The language is **total** by construction (DAG call graph + finite `foreach`),
  so the executor-in-the-loop always finishes.
- LLM paraphrasing is **pluggable and optional**; paraphrases are kept only if
  they pass **back-translation** (regenerate code, check behavioral equivalence).
- See `SPEC.md` for the grammar and `DATASET_SCHEMA.md` for the record schema.
