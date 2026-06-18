# Loom Dataset Schema v0.1.0

Loom writes streamed, sharded **JSON Lines** (`luc-NNNNN.jsonl`) plus a
`manifest.json`. JSON is emitted deterministically (sorted keys, compact
separators, ASCII), so identical `(base_seed, generator_version,
language_version, config)` reproduces identical bytes (PRD §8.5).

## Record fields

| field | type | meaning |
|---|---|---|
| `id` | string | `luc-000123`, sequential over accepted records |
| `language_version` | string | Lucid version that defines validity/meaning |
| `generator_version` | string | Loom version |
| `seed` | int | per-example derived seed (reproduces this record) |
| `curriculum_stage` | int | 0 (easy) … 5 (full features) |
| `difficulty` | int | 1–10 heuristic from size/depth/features |
| `program_canonical` | string | the program in canonical form |
| `ast_hash` | string | `sha256(program_canonical)`; dedup + provenance |
| `type_signature` | string | entry signature, e.g. `(#List[#Int]) -> #Int` |
| `io_examples` | list | `[{ "input": [...], "output": ... }]` (executed) |
| `spec_structured` | string | deterministic templated NL description from the AST |
| `spec_nl_paraphrase` | string? | optional LLM paraphrase (present only if a paraphraser is plugged in) |
| `spec_verified` | bool | paraphrase passed back-translation behavioral check |
| `features` | list[string] | feature tags (`foreach`, `record`, `builtin_add`, …) |
| `trace_available` | bool | whether a step trace is included |
| `trace` | object? | `{ input, output, steps, trace: [events] }` |
| `completion_pairs` | list? | `completion` (prefix→completion) and `infill` (prefix/middle/suffix) |
| `repair_pairs` | list? | `repair_type` (broken, error, fixed) and `repair_behavior` (broken, fixed, diff_input, outputs) |
| `refactor_pairs` | list? | `{ transform, a, b, relation: "equivalent" }`, behaviorally verified |
| `pair_types` | list[string] | which pair types this record carries |

`pair_types` is drawn from `spec_to_code`, `io_to_code`, `completion`, `trace`,
`repair`, `refactor`.

## Value encoding (IO / traces)

`#Int`→number, `#Bool`→true/false, `#Str`→string, `#List`→array,
record→`{"@record": name, "@fields": {…}}`, variant→`{"@variant": name,
"@tag": tag, "@args": […]}`, function value→`{"@fn": name}`.

## manifest.json

```jsonc
{
  "language_version": "0.1.0",
  "generator_version": "0.1.0",
  "base_seed": 0,
  "config_fingerprint": "…",
  "use_curriculum": true,
  "n_requested": 1000,
  "n_written": 1000,
  "attempts": 1003,
  "typecheck_bugs": 0,            // MUST be 0
  "discarded_no_io": 3,
  "coverage": {
    "production_coverage": 1.0,   // fraction of grammar productions exercised
    "missing_productions": [],
    "type_interactions": [ … ],
    "diversity_ratio": 1.0,       // distinct-canonical-AST ratio over sampled
    "unique": 1000, "total_seen": 1003
  },
  "shards": [ { "name": "luc-00000.jsonl", "count": 1000 } ],
  "total_records": 1000
}
```
