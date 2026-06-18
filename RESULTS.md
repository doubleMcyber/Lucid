# Experiment Results — can a small model learn Lucid?

**Question (PRD §4, M6):** can a small model emit *valid, executable* Lucid after
a modest single-machine fine-tune, and does fine-tuning on Loom data beat the
un-tuned base model? Measured exactly by the Lucid toolchain — no human grading.

## Setup

- **Model:** `Qwen/Qwen3-0.6B-Base` (~0.6B params), Apple Silicon **MPS**.
- **Fine-tune:** LoRA (r=16, α=32, all-linear), **1500 steps**, batch 4,
  grad-accum 1, lr 3e-4, **bf16**, max_len 192. ~10.1M trainable params, ~5.9k
  usable examples (≤192 tokens), ~40 min wall on MPS. Train loss 2.4 → 0.30.
- **Data:** Loom-generated *compact, single-function* Lucid programs (integers,
  booleans, lists; arithmetic, comparisons, `foreach`, `if`, `cond`) — 6000
  programs → 10,218 train / 1,782 test SFT pairs across two tasks, split
  leakage-free by `ast_hash`.
- **Tasks:** `spec_to_code` (NL spec + signature → program) and `io_to_code`
  (input/output examples + signature → program).
- **Eval:** greedy decoding, 60 held-out items per task. The generated program
  is parsed, type-checked, and executed against the held-out IO by the toolchain.

## Results

| Task | metric | base | fine-tuned |
|---|---|---|---|
| `spec_to_code` | parse_rate     | 0.00 | **1.00** |
| `spec_to_code` | typecheck_rate | 0.00 | **0.78** |
| `spec_to_code` | exec_pass@1    | 0.00 | **0.23** |
| `spec_to_code` | exact_match    | 0.00 | 0.05 |
| `io_to_code`   | parse_rate     | 0.00 | **1.00** |
| `io_to_code`   | typecheck_rate | 0.00 | **0.98** |
| `io_to_code`   | exec_pass@1    | 0.00 | **0.37** |
| `io_to_code`   | exact_match    | 0.00 | 0.03 |

## Interpretation

- **H1 (well-formedness) — strongly supported.** The base model produces **0%**
  parseable Lucid (it has never seen the language). After a ~40-minute LoRA
  fine-tune the model produces parseable Lucid **100%** of the time and
  type-correct Lucid **78–98%** of the time. The language is *learnable*, and
  fine-tuning on Loom data is the difference between "never valid" and "always
  parseable, usually type-correct."
- **H2 (functional reliability) — supported.** Executed pass@1 goes from **0%**
  to **23% (spec→code) / 37% (io→code)**. For `io_to_code` the model is doing
  genuine program-synthesis-from-examples: it reads the IO, infers the function,
  and emits code that *runs and reproduces the held-out outputs* a third of the
  time — and **everything it emits parses**, so each attempt is cheaply
  verifiable and safe to execute in a loop (the agent-target value proposition).
- **Why exec/exact are lower than typecheck.** The templated `spec` and a handful
  of IO examples *underdetermine* the exact reference program — many distinct
  Lucid functions satisfy them. So exact-match is near zero by design, and
  exec_pass@1 measures the harder bar of reproducing the reference's behavior
  exactly. `io_to_code` > `spec_to_code` on exec precisely because examples
  constrain behavior more than a templated description.

### A real fine-tuned generation (held-out `io_to_code`)

The prompt's examples imply *output = the second argument*; the model writes a
valid, type-correct, executable solution:

```
### Examples (input -> output):
  [[-20, -7, 19, -16], []] -> []
  [[5, -11, 17, -9], [false]] -> [false]
  [[0, -14], [true, false, false, false, false]] -> [true, false, false, false, false]
### Signature:
(#List[#Int], #List[#Bool]) -> #List[#Bool]
### Lucid:
fn @main ($v0 : #List[#Int], $v1 : #List[#Bool]) -> #List[#Bool] = do
  return $v1 ;
end @main ;
```
→ parses ✓, type-checks ✓, reproduces all held-out IO ✓.

## Caveats & what this is *not*

- This tests **H1/H2 on a compact subset** of Lucid. It does **not** yet test
  **H3** (data efficiency) or **H4** (small-vs-large gap), nor a **matched
  baseline language** (Python) head-to-head — those need the baseline pipeline in
  `IMPROVEMENTS.md §B` and are future work. The eval harness is baseline-ready.
- Compact programs were used so sequences fit a 0.6B model on a single Mac
  ("overthrottled"). Full-feature programs (records/variants/match/HOF) are
  longer; learnability there is expected but untested at this budget.

## Engineering notes (MPS)

- Qwen3 uses **grouped-query attention** (16 query / 8 KV heads). The optimized,
  cached attention path **miscomputes shapes on MPS during generation**
  (`mps.matmul` shape error). Fix: load with `attn_implementation="eager"`.
  Training (full-sequence) is unaffected.
- batch 8 at fp32 drove the machine into **17 GB of swap** (≈40 s/step). batch 4
  + bf16 + max_len 192 keeps it resident (~1–2 s/step). HF hub calls must be
  forced offline (`HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_XET=1`) or model loads hang.

## Reproduce

```bash
export SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())')
# one-time: cache the weights (xet disabled to avoid a hang)
HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import hf_hub_download as d; d('Qwen/Qwen3-0.6B-Base','model.safetensors')"
# prepare data, train (MPS, bf16), evaluate base vs tuned
LUCID_DTYPE=bf16 LUCID_MAXLEN=192 python scripts/experiment.py prepare --n 6000
LUCID_DTYPE=bf16 LUCID_MAXLEN=192 python scripts/experiment.py train --steps 1500 --batch 4 --grad-accum 1 --lr 3e-4 --lora-r 16
LUCID_DTYPE=bf16 python scripts/experiment.py report --limit 60
```
