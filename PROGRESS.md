# PROGRESS — Lucid / Loom

Living status doc. **Updated after every goal cycle.** Newest cycle on top of the
log. For the *why* and *what-next* see `IMPROVEMENTS.md`; for headline numbers see
`RESULTS.md` (standalone H1/H2) and `BASELINE.md` (matched Lucid-vs-Python).

## What this project is

A falsifiable language-design thesis (PRD): *syntax designed for sequence models
improves reliability and data-efficiency.* Three artifacts:

- **Lucid** — a total, canonical, bijective, executable language (`lucid/`).
- **Loom** — a type-directed synthetic-data engine + eval harness (`loom/`).
- **Experiments** — small-model fine-tunes that measure H1–H4 (`scripts/`, `experiment/`).

## Hypotheses

- **H1** well-formedness (parse/typecheck rate) — *standalone: supported.*
- **H2** executed pass@1 vs held-out IO — *standalone: supported.*
- **H3** data-efficiency (accuracy vs #examples) — *partial: Lucid more efficient
  at n≤1000 (+0.10/+0.05), Python passes at 2000 (−0.08); within n=60 noise.*
- **H4** small-vs-large-model gap — *not started.*

## Status board

| Track | Item | State |
|---|---|---|
| A | Dataset exporter, eval harness, tokenizer analysis | ✅ done |
| A | Qwen3-0.6B LoRA fine-tune + base-vs-tuned A/B (`RESULTS.md`) | ✅ done |
| A | Eval harness adversarially audited + hardened | ✅ done |
| B | Execution-feedback eval (pass@1 vs hidden IO) | ✅ done |
| B | Repair training signal with real errors | ✅ done |
| B | Matched **Python baseline** pipeline (transpile, sandbox, eval) | ✅ done |
| B | **Matched H1–H3 sweep** (`compare.py`, Lucid vs Python) | ✅ done (v2) |
| B | Grammar/type-constrained decoding | ⬜ roadmap |
| C | Feature-flag ablation, difficulty grading, diversity metric | ✅ done |
| C | Naturalness (real task/spec distributions) | ⬜ roadmap |
| C | Real-LLM paraphrase (back-translation verified) | ⬜ roadmap |
| D | Generics / bounded recursion / human surface | ⬜ roadmap (non-goals for now) |

## Active task: matched Lucid-vs-Python sweep (H1–H3)

`scripts/compare.py` trains the SAME base model on the SAME programs in two
surface languages (Python transpiled from the identical Loom programs, identical
split) and compares well-formedness + executed pass@1 at matched data/compute.

- **Unfair sweep** (raw-capped, 750/1500/3000): done → `results_unfair.jsonl`.
  Verdict: confounded — Lucid is more verbose, so a raw cap + length filter
  silently gave Python more usable examples and more optimizer steps.
- **Fair sweep** (cap on *usable* ≤max_len examples, 500/1000/2000): the fix that
  equalizes data + steps. **This is the run in flight.**

## Goal cycle log

### Cycle 2 — 2026-06-18 — Fix swap-thrash; relaunch fair sweep
**Goal:** diagnose the 15-hour "throttling", fix it, resume the fair H1–H3 sweep.

**Diagnosis (evidence-backed):** not throttling — **swap thrash**. `vm.swapusage`
14.8/16 GB; step time crept 12→74 s/step *within* each cell (allocator growth),
reset fast each new cell (rules out thermal); the first run hit 1.12 s/step on a
fresh machine. Root cause = **fp32 + max_len 384 + no `torch.mps.empty_cache()`
in the train loop** on a 32 GB Mac whose RAM was already eaten by VS Code/Chrome
plus the prior run's leaked pages (swap had ballooned to ~16 GB). The run had
already crashed (leaked-semaphore) at `python|1000`.

**Fix:**
- `compare.py::_train`: free per-step tensors (`del out, ii, am, lb`) + flush the
  MPS allocator every 25 steps. Stops the monotonic per-step creep.
- `compare.py::_eval`: release MPS cache after each task's generation.
- Record `dtype/batch/grad_accum/lr/lora_r` in each result row (was missing —
  reproducibility hole).
- Relaunch with `LUCID_DTYPE=bf16` (2× headroom) + `--batch 2 --grad-accum 2`
  (half activation peak, same effective batch). `max_len` kept at **384** to
  preserve the matched-program invariant (≥99% of both languages fit).
- Archived the 3 partial fair cells (batch-4/fp32 config) → rerun all 6 clean.

**Hardening (from an adversarial review run before relaunch — no blockers, MPS
fix validated):** also fixed 4 real findings in `compare.py` so the rerun is the
definitive one:
- **Matched-program invariant made true.** `_build_examples` now draws from the
  `(id,task)` intersection usable in *both* languages via one seed-shuffled key
  list → identical programs *and* counts (was: per-language length filter dropped
  different rows; ~0.5% divergence at max_len 384, 18% at 192). Verified: 10,084
  shared-usable programs; n=500/1000/2000 yield identical program sets.
- **Generation budget 140 → 384.** 140 truncated correct Lucid more often than
  Python (Lucid completion p99=299/max=352 vs Python 293/337) — an unmatched eval
  budget. 384 clears both maxima.
- **Crash-proof results.jsonl.** `_read_rows()` skips truncated/malformed lines
  (a killed process left an un-resumable file before) and dedups by key last-wins
  (report had used the stale first row on rerun).
- **Honest caveats baked into `BASELINE.md`** (regenerated each run): the matched
  setup + 3 residual asymmetries, all *conservative against Lucid* (typecheck-gated
  exec, apples-to-oranges 2nd metric, JSON-string vs `==` output equality).

Validated end-to-end with a 2-cell smoke (217s, no thrash, no crash).

**Status: ✅ complete.** Sweep ran 6 cells in ~1h50m at a flat **0.65–0.75 s/step**
(was 12–74 and climbing → swap-thrash gone, confirmed not thermal). Result below.

**Verdict (H1/H3, exec@1, matched programs + data + compute):**

| n_train | Lucid | Python | Δ (L−P) |
|---|---|---|---|
| 500  | 0.292 | 0.192 | **+0.100** |
| 1000 | 0.283 | 0.233 | **+0.050** |
| 2000 | 0.383 | 0.467 | **−0.083** |

- **H1 (well-formedness):** both near-100% parse + high typecheck/runs after FT;
  Lucid marginally more reliable when data-starved (Python parse dips to 0.82–0.87
  at n≤1000). Supported, ≈tie.
- **H3 (data-efficiency):** *directional, not decisive.* Lucid leads when data is
  scarce (≤1000) and Python overtakes by 2000 — i.e. Lucid's edge is a
  low-data-efficiency effect that erodes with scale. **All three deltas are within
  ~1–2× the n=60-per-task sampling error (SE≈0.06 on a delta), so this is
  suggestive, not significant.** The residual asymmetries are all conservative
  *against* Lucid, so its small-N edge is if anything understated.
- **Honest takeaway:** the strong "Lucid ≫ Python" headline is **not** supported at
  matched compute on this compact subset; the defensible claim is "competitive,
  with a small data-efficiency edge at low N that needs a larger eval to confirm."
- **Next to firm this up:** raise eval `limit` (60→200+) to shrink the error bars;
  add more/again-smaller n_train points (e.g. 250) to trace the efficiency curve.

### Cycle 1 — 2026-06-17 — Matched Python baseline + comparison driver
**Goal:** make H1–H4 testable head-to-head against Python at matched data/compute.
**Done:** Python transpiler from Loom, sandboxed Python eval, leak auditor on both
splits, `compare.py` driver (H1–H4), matched-corpus test. Committed (`c78b59e`).

### Cycle 0 — 2026-06-17 — Standalone H1/H2 + audit
**Goal:** can a 0.6B model emit valid, executable Lucid after a modest fine-tune?
**Done:** base 0% valid → tuned 100% parse / 80–98% typecheck / 25–43% exec@1.
Harness adversarially audited (empty-output, prompt leakage, shown-example
scoring) and re-measured. See `RESULTS.md`.

## Last run

- **Sweep:** fair v2 — bf16, batch 2 / accum 2, max_len 384, sizes 500/1000/2000,
  epochs 2, limit 60. Ran 2026-06-18 10:54→12:44 (~1h50m), exit 0, all 6 cells.
  Log `experiment/compare/sweep_fair_v2.log`; data `results.jsonl`; numbers in
  `BASELINE.md`. Config: `scripts/run_fair_sweep.sh`. Old batch-4/fp32 partials
  archived to `results_fair_partial_batch4.jsonl`.
- **Throttle:** flat **0.65–0.75 s/step** start to finish (was 12–74 and climbing).
  Fixed and confirmed.
- **Headline:** Lucid +0.100 / +0.050 / −0.083 exec@1 vs Python at n=500/1000/2000
  (within n=60 noise — see Cycle 2 verdict).

## Next actions

1. **Shrink the error bars** on the H3 verdict: rerun `compare.py report`-style
   eval with `--limit 200` (n=60→200/task) and add an n=250 point, so the
   small-N data-efficiency edge is either confirmed or refuted out of noise.
2. **(H4)** add a second, larger base model to the sweep for the small-vs-large gap.
3. **(B)** prototype the generate→parse→typecheck→repair constrained-decoding wrapper.
