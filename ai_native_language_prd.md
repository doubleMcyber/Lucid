# PRD: Lucid — A Model-Native Language and the Loom Corpus Generator

**Status:** Draft v0.1 · **Owner:** _[you]_ · **Last updated:** 2026-06-17

> Working codenames: **Lucid** (the language) and **Loom** (the corpus generator). Both are placeholders.

---

## 1. Summary

Lucid is a programming language designed from scratch around a single constraint that no mainstream language was designed for: it must be easy for a sequence model to read and write *correctly*, token by token, with minimal long-range bookkeeping and zero local ambiguity. Loom is the tool that gives the language reach — a constrained program synthesizer that procedurally generates large volumes of valid Lucid programs paired with specifications, input/output examples, execution traces, and repair/refactor variants, suitable for training and evaluating code models.

The grammar, parser, type checker, and interpreter are deliberately small. The synthesizer is the bulk of the work and the bulk of the value: it is a type-directed generator with a semantic-validation loop and a set of labelers that turn each generated program into multiple kinds of training pairs. Total scope is on the order of 10K lines.

This PRD treats the language as a **falsifiable hypothesis** — "syntax designed for sequence models measurably improves model reliability and data-efficiency at code tasks" — and is structured so that Loom and the evaluation harness are useful even if that hypothesis turns out to be weak.

---

## 2. Problem and motivation

Code is among the highest-value domains for language models, yet every language a model writes today was designed for human ergonomics, human editors, and 1970s–2010s parser theory. Models therefore spend representational capacity, and make avoidable errors, on syntactic features that exist for reasons unrelated to them:

- **Ambiguous and context-dependent tokenization.** Where a token boundary falls can depend on surrounding context. Models must learn lexing as a side task.
- **Long-range structural dependencies.** A closing brace must match an opener that may be hundreds of tokens away. Balanced-delimiter tracking and precedence resolution are exactly the kind of counting/matching that transformers do unreliably.
- **Significant whitespace.** Indentation-as-syntax (Python) forces a model to emit exact, consistent leading whitespace it cannot locally verify.
- **Optional and elidable syntax.** Optional semicolons, implicit returns, inferred types, and "there are six idiomatic ways to write this" spread the target distribution across stylistic variants the model must learn to both produce and accept.
- **Overloaded tokens.** A single symbol (`*`, `&`, `<`) carries multiple meanings disambiguated only by distant context.

The hypothesis is that a language deliberately stripped of these properties lets a *smaller* model be *more reliable*, makes model output trivially verifiable (because it is canonical, executable, and round-trippable to an AST), and — critically for agents — gives an agent that writes-and-runs code a target language whose output is guaranteed parseable and easy to check.

A second, independent motivation: **synthetic data.** Whatever one believes about the language, a generator that emits unlimited, fully-labeled, executable, diverse programs is a useful artifact for training and evaluating code models in a controlled setting, and a clean testbed for the research question "how does language design affect model performance?"

### Why now

Code models are good enough that the marginal gains from raw scale are getting expensive, attention is shifting to data quality and verifiability, and agentic systems increasingly emit code that is executed automatically — which makes guaranteed-parseable, easy-to-verify output economically valuable rather than merely elegant.

---

## 3. Goals and non-goals

### Goals

1. Specify and implement a small language whose surface form is locally decodable, whitespace-insensitive, free of overloaded tokens, and has a single canonical form per program with a bijective surface↔AST mapping.
2. Build Loom: generate large volumes of valid, diverse, difficulty-graded Lucid programs with near-100% validity by construction, fully deterministic given a seed.
3. Produce multiple training-pair types per program (spec→code, IO→code, completion, trace, repair, refactor) under a documented schema.
4. Build an evaluation harness that tests the core hypotheses against a real-language baseline.
5. Keep the whole system reproducible, versioned, and around 10K lines.

### Non-goals (v1)

- **Not** a language for humans to enjoy writing. Terseness and human ergonomics are explicitly sacrificed; a friendlier human-facing surface is a future extension, not v1.
- **Not** a production compiler. No optimization, no native codegen, no FFI; the interpreter exists to validate semantics and produce IO/traces.
- **Not** Turing-complete in v1. The language is deliberately **total** (guaranteed-terminating) so the executor-in-the-loop never hangs. General recursion is a future extension behind resource limits.
- **Not** a claim that the language beats training-more-on-Python. That is the experiment, not an assumption.
- **Not** dependent on a specific foundation-model vendor; any LLM use (optional NL paraphrasing) is pluggable and validated.

---

## 4. Hypotheses and success metrics

The product exists to test falsifiable claims. Each has a metric and a baseline (the same base model, matched compute/data, emitting a mainstream language — Python or a comparably-featured subset).

- **H1 — Well-formedness.** A model fine-tuned to emit Lucid produces parseable *and* type-correct output at a higher rate than the baseline. *Metric:* % of generations that parse and typecheck, at matched task difficulty.
- **H2 — Functional reliability.** On synthesis-from-spec, executed pass@1 is higher in Lucid than baseline. *Metric:* pass@k against held-out specs, measured by running the program against hidden IO.
- **H3 — Data efficiency.** Reaching a target accuracy requires fewer training examples in Lucid. *Metric:* accuracy-vs-data-volume curves.
- **H4 — Small-model viability.** The large-vs-small model gap is narrower in Lucid (the language buys reliability that otherwise costs scale). *Metric:* accuracy delta across two model sizes, Lucid vs. baseline.

**Generator-level metrics (success regardless of H1–H4):**

- Validity rate ≥ 99.9% (target ~100% by construction; failures indicate generator/typechecker bugs).
- Throughput: ≥ 1,000 fully-labeled examples/sec/core for simple programs (order-of-magnitude target, to be tuned).
- Diversity: distinct-canonical-AST ratio ≥ 0.95 within a shard; grammar-production coverage ≥ 99%; type-interaction coverage tracked and reported.
- Pairing yield: ≥ 1 verified spec pair per program (templated), with optional LLM paraphrase yield reported separately.
- Reproducibility: identical dataset bytes from identical (seed, generator-version, language-version).

---

## 5. Users

- **ML researchers studying code models** who want a controlled environment to isolate the effect of language design, and a configurable synthetic-data source. *Primary.*
- **Agent builders** who want a codegen target language they fully control, so model output is guaranteed parseable and cheap to verify and execute in a loop. *Primary, and the most likely near-term commercial wedge.*
- **You, as a data/model-research engine** — a self-contained pipeline to generate data, fine-tune, evaluate, and iterate on language design. *Internal.*

---

## 6. Product overview

Two components with a clean seam between them:

1. **Lucid toolchain** — lexer, parser, canonical printer (bijective with the AST), type checker, and a total interpreter. Small, correct, heavily property-tested. This defines and enforces the language.
2. **Loom corpus generator** — consumes the toolchain as a library. A type-directed AST sampler produces well-typed programs; a validation loop typechecks and executes them; a bank of labelers turns each into training pairs; a coverage/diversity controller shapes the distribution and curriculum; a writer streams a sharded, schema'd, reproducible dataset.

The seam matters: the toolchain is the *specification of truth* (what is a valid program, what does it mean), and Loom is everything that exploits it. The toolchain is ~30% of the code; Loom is ~70%.

---

## 7. Language design requirements (Lucid)

These are concrete, opinionated decisions for v1. Where a decision is genuinely open, it is marked.

### 7.1 Tokenization and lexical design

- **No context-dependent lexing.** A token's lexical class is determined by its own characters, never by surrounding context.
- **Whitespace is non-semantic.** Indentation and newlines carry no meaning; the canonical printer normalizes them, and the parser ignores them. A model never has to emit exact whitespace to be correct.
- **Reserved, fixed-form keywords and no infix operators.** All operations are prefix named calls — `add($a, $b)`, not `$a + $b`. This eliminates operator-precedence resolution (a long-range, ambiguity-prone task) entirely and makes every expression a uniform call tree. *(Opinionated; the single most distinctive choice.)*
- **Tokenizer-aware vocabulary.** Keywords and built-in names are chosen to be single tokens under common BPE tokenizers where feasible. *(Open: degree of co-design with a specific tokenizer vs. robustness across tokenizers.)*
- **Escape-free string literals.** String literals use a self-delimiting fenced form that cannot collide with their own contents, removing in-band escaping (a classic source of model errors). *(Open: exact fencing scheme.)*

### 7.2 Structure and delimiters

- **No optional syntax, anywhere.** Every statement has a mandatory terminator; every block has explicit open and close; nothing is elidable.
- **Self-naming block terminators.** A block closes with a token that names what it closes — `end @sum`, `end foreach` — so structure is locally self-correcting and the model never has to match an anonymous closer to a distant opener.
- **Single canonical form.** Exactly one canonical text exists per AST; a normalizer enforces it. There are no stylistic variants, so the training target distribution is not spread across formatting choices.
- **Bijective surface↔AST.** The surface form is a deterministic serialization of the AST and parses back to the identical AST. This is the property that makes Loom possible: generate ASTs, print canonically, get perfect labels for free.

### 7.3 Identifiers, kinds, and local type information

- **Kind sigils.** Identifiers carry a sigil encoding their kind: `$` local, `@` function, `#` type, `%` field. A model can resolve an identifier's kind from the token itself, without tracking distant declarations.
- **Explicit types everywhere in canonical form.** Every binding and parameter is annotated. Local type information is always present in the token stream, attacking the long-range-dependency weakness directly. (A future ergonomic surface may infer types and elaborate to the fully-annotated canonical form.)

### 7.4 Type system

- Static, strong, with no implicit coercion, no subtyping, and no ad-hoc overloading in v1.
- Base types plus records, tagged variants, fixed homogeneous lists/arrays, and first-class functions. Parametric generics are a likely v1.x extension; kept out of v1 to keep generation tractable.
- Strong typing is load-bearing three times over: it powers the type-directed generator, it makes programs verifiable, and it puts type info locally in front of the model.

### 7.5 Semantics and runtime

- **Total by construction.** All iteration is bounded (e.g., `foreach` over a finite structure; structural recursion only). Every program terminates, so the executor-in-the-loop always finishes — a large practical win for generating IO pairs at scale.
- **Deterministic, effect-controlled.** No ambient I/O; a program is a pure function from inputs to output, with an optional controlled effect set later. Every program has a definite result on given inputs, enabling reliable IO pairing.

### 7.6 Illustrative canonical snippet

```
fn @sum ($xs : #List[#Int]) -> #Int = do
  var $acc : #Int = 0 ;
  foreach $x : #Int in $xs do
    set $acc = add($acc, $x) ;
  end foreach ;
  return $acc ;
end @sum ;
```

Every token's role is locally determined: sigils give kind, types are explicit, the block self-closes by name, statements are explicitly terminated, all operations are named prefix calls with no precedence, and the whitespace shown is decorative.

---

## 8. Corpus generator requirements (Loom)

This is the bulk of the system. Pipeline: **sample → validate → label → shape → write.**

### 8.1 AST sampler (the heart)

- **Type-directed, top-down sampling.** Given a target type and a size budget, sample a well-typed AST by applying the typing rules in reverse, drawing from in-scope bindings, declared functions, and constructors. Programs are well-typed *by construction*, not by rejection.
- **Scope/context manager.** Tracks in-scope bindings so every reference is valid; generates declarations before uses; respects sigil/kind rules.
- **Controls.** Size distribution, per-construct frequency weights, maximum nesting depth, and feature flags selecting which language features are in play (for curricula).

### 8.2 Semantic validator

- Typecheck (should always pass; a failure flags a generator bug and is logged loudly).
- Execute via the interpreter to obtain the result and, optionally, a full step trace. Enforce resource bounds on size/steps. Discard anything that errors.

### 8.3 Labelers (pairing engines)

Each accepted program yields several training pairs:

- **IO examples** — run on sampled inputs; record (input, output).
- **Execution trace** — step-by-step trace for trace/explain training.
- **Completion / fill-in-the-middle** — prefix→continuation and span-infill pairs, made clean by the canonical form.
- **Repair** — inject a typed mutation that breaks typecheck or changes behavior; pair (broken, error/diff, fixed).
- **Refactor / equivalence** — apply a semantics-preserving transform (rename, inline, reorder); pair (A, B, "equivalent").
- **Specification** — a deterministic, templated NL/structured description derived from the AST; *optionally* an LLM paraphrase that is **validated by back-translation** (regenerate code from the paraphrase, check behavioral equivalence; keep only verified pairs).

### 8.4 Coverage, diversity, and curriculum control

- **Coverage-guided sampling** (fuzzer-style): track grammar-production and type-interaction coverage; bias toward under-covered regions.
- **Dedup** on canonical-AST hash to avoid near-duplicates.
- **Curriculum:** difficulty scoring (by size, depth, feature set, branching) and a schedule that emits an easy→hard progression.

### 8.5 Output and reproducibility

- Streamed, sharded JSONL under a documented schema.
- Every record carries provenance: seed, generator version, language version, AST hash. Identical inputs reproduce identical bytes.

### 8.6 Sample training record

```json
{
  "id": "luc-000123",
  "language_version": "0.1.0",
  "generator_version": "0.3.1",
  "seed": 8472,
  "program_canonical": "fn @sum ($xs : #List[#Int]) -> #Int = do ... end @sum ;",
  "ast_hash": "b3f1c9...",
  "type_signature": "(#List[#Int]) -> #Int",
  "io_examples": [
    { "input": [[1, 2, 3]], "output": 6 },
    { "input": [[]], "output": 0 }
  ],
  "spec_structured": "Return the sum of a list of integers.",
  "spec_nl_paraphrase": "Add up all the numbers in the given list and return the total.",
  "spec_verified": true,
  "trace_available": true,
  "difficulty": 2,
  "features": ["foreach", "mutable_local", "builtin_add"],
  "pair_types": ["spec_to_code", "io_to_code", "completion", "trace"]
}
```

---

## 9. System architecture and LOC budget

Data flows one direction: toolchain primitives → sampler → validator → labelers → shaper → writer. The validator and every labeler call back into the same parser/printer/typechecker/interpreter, so there is exactly one source of truth for what a valid program is and what it means.

Indicative line-count allocation (validates the ~10K estimate and shows the bulk sits in the generator):

| Component | Approx. LOC |
|---|---|
| Lexer + parser + canonical printer (bijective) | 1,000–1,500 |
| Type checker | 1,000–1,500 |
| Interpreter (total semantics) + tracer | 1,000–1,500 |
| AST/type-directed sampler + scope manager | 2,000–2,500 |
| Labelers (IO, trace, completion, repair, refactor, spec) | 2,000–2,500 |
| Coverage / diversity / curriculum control | 500–1,000 |
| Dataset writer, CLI, config, reproducibility plumbing | 500–1,000 |
| **Total** | **~9,000–11,000** |

The sampler plus labelers — the corpus generator proper — are roughly 60–70% of the code, exactly as the original framing predicted.

---

## 10. Milestones

A roughly one-year, milestone-driven plan ending in the falsifiable test.

- **M0 — Spec (weeks 1–3).** Settle §7 decisions, write the formal grammar, define the canonical form, hand-author ~20 example programs. *Deliverable:* language spec + grammar.
- **M1 — Front end (weeks 4–8).** Lexer, parser, canonical printer. *Deliverable:* round-trip property tests pass on all examples (parse→print→parse is identity).
- **M2 — Meaning (weeks 9–14).** Type checker + total interpreter + tracer. *Deliverable:* all examples typecheck and execute; property tests for well-typed→no runtime type error and guaranteed termination.
- **M3 — Core generator (weeks 15–22).** Type-directed sampler, scope manager, validation loop. *Deliverable:* generate N valid, diverse, deterministic programs with IO pairs; validity ~100%; first coverage report.
- **M4 — Labelers + dataset (weeks 23–30).** All pair types, templated specs, schema, sharded writer, provenance. *Deliverable:* a v1 dataset (target ~1M examples across pair types).
- **M5 — Shaping + scale (weeks 31–40).** Coverage/diversity/curriculum control, throughput work, optional validated LLM paraphrasing. *Deliverable:* curriculum'd, deduped, coverage-targeted dataset at scale.
- **M6 — Evaluation (weeks 41–52).** Fine-tune small models on Lucid vs. a matched baseline; run H1–H4; write up. *Deliverable:* experimental report and a go/no-go judgment on the language thesis.

---

## 11. Risks and open questions

- **The thesis may be dominated by scale.** Maybe more data on existing languages wins and language design contributes little. *Mitigation:* structure the project so Loom (a synthetic-data engine) and the eval harness (a clean testbed) are valuable regardless; treat the language as a hypothesis, not a bet.
- **Transfer/distribution gap.** Models trained on synthetic Lucid may not transfer to tasks people care about. *Mitigation:* position Lucid first as an agent-target language (where you control the output language anyway), and measure transfer explicitly rather than assuming it.
- **Tokenizer dependence.** "Single-token keywords" depends on a tokenizer; the design may need to co-design with one or prove robustness across several. *Open.*
- **Spec quality vs. dependency.** Templated NL is low-diversity; LLM paraphrase adds a dependency and noise. Back-translation validation helps but costs compute. *Open.*
- **Diversity vs. validity tension.** Generation that guarantees validity tends toward a narrow distribution; pushing diversity risks invalid or unnatural programs. This is the central engineering tension in Loom.
- **Naturalness.** Synthetic programs may be structurally unlike real human-intent code (no real domains, libraries, or idioms), limiting usefulness for general codegen. *Mitigation:* later, drive generation from real task/spec distributions rather than pure grammar sampling.
- **Canonical vs. usable.** The most model-friendly form may be unusable by humans; a dual-surface design could reintroduce the ambiguity we removed. *Open; deferred past v1.*

---

## 12. Future directions

- An ergonomic human-facing surface that elaborates to canonical Lucid (terse to write, canonical to train on).
- A transpiler bridging a mainstream language to Lucid, to study whether reliability gains survive translation.
- Verification hooks: because programs are total, canonical, and executable, attach property checks or lightweight proofs to generated programs.
- A self-improvement loop: generate → train → use the improved model to drive harder, more natural generation → repeat.
- Integration as the codegen target in an agent runtime, where guaranteed-parseable, easy-to-verify output is the point.

---

## 13. Appendix: definitions

- **Canonical form** — the unique surface text the printer emits for a given AST.
- **Bijective surface↔AST** — printing then parsing (and vice versa) is the identity map.
- **Total language** — every program provably terminates.
- **Type-directed generation** — sampling well-typed ASTs by applying typing rules in reverse, rather than generating then filtering.
- **Back-translation validation** — accepting an NL paraphrase only if regenerating code from it reproduces the original program's behavior.
