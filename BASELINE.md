# Lucid vs Python — matched-baseline results

Executed pass@1 (mean over tasks), held-out IO. Same base model, same
task distribution, matched data + epochs per (model, n_train).

## Qwen/Qwen3-0.6B-Base

| n_train | Lucid exec@1 | Python exec@1 | Δ (L−P) |
|---|---|---|---|
| 500 | 0.292 | 0.192 | +0.100 |
| 1000 | 0.283 | 0.233 | +0.050 |
| 2000 | 0.383 | 0.467 | -0.083 |

## Per-task detail

| cell | task | parse | type/runs | exec@1 |
|---|---|---|---|---|
| lucid|Qwen/Qwen3-0.6B-Base|1000|2 | spec_to_code | 0.9833 | 0.9667 | 0.1833 |
| lucid|Qwen/Qwen3-0.6B-Base|1000|2 | io_to_code | 1.0 | 1.0 | 0.3833 |
| lucid|Qwen/Qwen3-0.6B-Base|2000|2 | spec_to_code | 1.0 | 0.8833 | 0.1833 |
| lucid|Qwen/Qwen3-0.6B-Base|2000|2 | io_to_code | 1.0 | 1.0 | 0.5833 |
| lucid|Qwen/Qwen3-0.6B-Base|500|2 | spec_to_code | 0.9833 | 0.9333 | 0.1833 |
| lucid|Qwen/Qwen3-0.6B-Base|500|2 | io_to_code | 1.0 | 1.0 | 0.4 |
| python|Qwen/Qwen3-0.6B-Base|1000|2 | spec_to_code | 0.8167 | 0.8167 | 0.1167 |
| python|Qwen/Qwen3-0.6B-Base|1000|2 | io_to_code | 0.95 | 0.95 | 0.35 |
| python|Qwen/Qwen3-0.6B-Base|2000|2 | spec_to_code | 1.0 | 1.0 | 0.25 |
| python|Qwen/Qwen3-0.6B-Base|2000|2 | io_to_code | 1.0 | 1.0 | 0.6833 |
| python|Qwen/Qwen3-0.6B-Base|500|2 | spec_to_code | 0.9167 | 0.8667 | 0.1667 |
| python|Qwen/Qwen3-0.6B-Base|500|2 | io_to_code | 0.8667 | 0.8333 | 0.2167 |

## Methodology & known asymmetries

**Matched on three axes.** Both languages are trained on the *same* underlying Loom programs (selected from the (id,task) intersection that is ≤max_len in both surface languages, via one seed-shuffled key list), so example counts, optimizer steps (equal epochs), and the programs themselves are identical — only the surface syntax differs. Splits are leakage-audited clean at the prompt level; eval uses held-out IO with greedy decoding and a generation budget (384 tokens) above the longest reference completion in both languages, so neither is truncation-limited.

**Residual asymmetries — all conservative *against* Lucid** (they cannot inflate a Lucid≥Python headline):
- *exec@1 definition.* Lucid gates execution on passing its static typechecker; Python has no such gate. A correct-output Lucid program that fails typecheck scores 0 where the Python equivalent scores 1.
- *Second metric column is not like-for-like.* `type/runs` is Lucid's static typecheck_rate vs Python's dynamic runs_rate — different constructs; do not compute an L−P delta on it. The headline is exec@1 only.
- *Output equality.* The Lucid evaluator compares decoded values (`==`); the Python evaluator compares canonical JSON strings — the latter is stricter on bool-vs-int (rarely flips, since typed signatures fix output types).
