# Lucid + Loom — Gap Analysis & Improvement Checklist

A standing analysis of where this project is, what it is *for*, and what would
make it genuinely useful. Checked items are implemented in this repo; unchecked
items are the prioritized roadmap.

## What is this project, really?

The PRD frames Lucid as a **falsifiable hypothesis** ("syntax designed for
sequence models measurably improves reliability and data-efficiency") plus two
artifacts that are valuable *regardless of whether the hypothesis holds*: a
synthetic-data engine (Loom) and an evaluation testbed.

### What the current architecture is best suited for

1. **An agent codegen target.** Lucid is total, canonical, bijective, and
   executable. An agent can *write → run → verify* in a loop with output that is
   guaranteed parseable and cheaply checkable. This is the strongest near-term,
   commercial-leaning use, and the architecture already supports it end-to-end.
2. **A controlled synthetic-data engine.** Type-directed generation gives
   ~100%-valid, fully-labeled, diverse, difficulty-graded, reproducible corpora
   — a clean instrument for studying how language design affects model behavior.
3. **An execution-feedback RL testbed.** Because every program is total and has a
   definite IO behavior, reward = "passes hidden IO" is cheap, safe, and
   non-hanging — ideal for RL-from-execution experiments.

### What AI needs for coding in the future (and how Lucid maps to it)

| Need | Lucid/Loom lever |
|---|---|
| Verifiable, executable output | total interpreter + IO pairs + canonical/bijective form |
| Reliability over cleverness | parse + typecheck guarantees; no ambiguity |
| Execution feedback for RL | total semantics → safe, fast reward signal |
| Spec→code with verification | templated spec + back-translation validation |
| Self-correction / repair | repair pairs with captured error messages |
| Small-model viability (cost) | the whole H4 thesis; testable here |
| Guaranteed-valid generation | (roadmap) grammar/type-constrained decoding |

## Checklist

### A. Make the thesis testable end-to-end (the payoff) — IMPLEMENTED
- [x] **Dataset exporter** (`loom/export.py`): Loom records → `(prompt, completion)`
      SFT pairs for `spec_to_code` / `io_to_code` / `completion`, with a
      deterministic train/test split and eval metadata (IO, signature, reference).
- [x] **Model eval harness** (`loom/evaluate.py`): model-agnostic; computes
      parse-rate, typecheck-rate, and executed **pass@1** vs held-out IO (H1/H2),
      plus a structural-validity baseline. `value_from_json` round-trips IO.
- [x] **Tokenizer-friendliness analysis** (`scripts/tokenizer_analysis.py`):
      measures tokens/program and tokens/char under a real BPE tokenizer
      (PRD §7.1 open question), Lucid vs. an equivalent Python rendering.
- [x] **Qwen3-0.6B LoRA fine-tune on MPS** + **base-vs-tuned A/B**
      (`scripts/experiment.py`) → `RESULTS.md` (PRD M6 / H1 / H2).
      **Result:** base 0% valid Lucid → fine-tuned **100% parse, 78–98%
      typecheck, 23–37% executed pass@1**. H1/H2 supported on a compact subset.

### B. Agent-usefulness — ROADMAP
- [x] Execution-feedback evaluation (pass@1 vs hidden IO) — in the eval harness.
- [x] Repair training signal with real error messages — captured in labelers.
- [ ] **Grammar/type-constrained decoding** so generation is valid *by
      construction at inference time* (the agent-target endgame). Lucid's
      declare-before-use + explicit types make a pure CFG grammar insufficient;
      a typed, stateful decoder (or a generate→validate→repair wrapper) is the
      right shape. A `generate→parse→typecheck→repair` loop is already expressible
      with the toolchain; full constrained decoding is future work.
- [ ] **Python (or Python-subset) baseline pipeline** for a head-to-head H1–H4
      comparison at matched compute/data (large; the eval harness is baseline-ready).

### C. Generator quality — IMPLEMENTED / ROADMAP
- [x] Correct feature-flag ablation (incl. `use_bool`/`use_lists`); genuinely
      simple curriculum stage 0; nested-list gating.
- [x] Difficulty grading that spans 1–10 and correlates with curriculum stage.
- [x] Diversity metric = distinct-AST ratio over written records (robust to
      `--no-dedup`).
- [ ] **Naturalness**: drive generation from real task/spec distributions rather
      than pure grammar sampling (PRD risk §11).
- [ ] **Real-LLM paraphrase**: wire a local model as the (already pluggable)
      paraphraser; keep only back-translation-verified pairs.

### D. Language/toolchain — ROADMAP (per PRD non-goals)
- [ ] Parametric generics (v1.x).
- [ ] Bounded structural recursion behind a fuel budget.
- [ ] An ergonomic human surface that elaborates to canonical Lucid.
