"""End-to-end learnability experiment (PRD M6 / H1 / H2):

Can a small model (Qwen3-0.6B) learn to emit *valid, executable* Lucid from a
modest, single-machine LoRA fine-tune on Loom-generated data — and does
fine-tuning on Loom data beat the un-tuned base model?

Subcommands:
  prepare  generate a compact Loom dataset and export train/test SFT pairs
  train    LoRA fine-tune Qwen3-0.6B on the train split (MPS)
  eval     evaluate a model (base, or base+adapter) on the test split
  run      do all of the above and write RESULTS.md

Designed to run on Apple Silicon (MPS) with small budgets ("overthrottled"):
short programs, LoRA, small batches + grad accumulation, capped eval set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Use only the local HF cache and avoid the xet backend — both have caused
# multi-minute hangs on hub metadata/weight calls. Weights must be pre-cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from loom.config import SamplerConfig  # noqa: E402
from loom.evaluate import evaluate  # noqa: E402
from loom.export import export_dataset  # noqa: E402
from loom.pipeline import GenSpec, generate  # noqa: E402

MODEL_ID = os.environ.get("LUCID_MODEL", "Qwen/Qwen3-0.6B-Base")
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "experiment")
RAW = os.path.join(WORK, "raw")
DATA = os.path.join(WORK, "data")
ADAPTER = os.path.join(WORK, "adapter")
MAX_LEN = int(os.environ.get("LUCID_MAXLEN", "256"))
TASKS = ["spec_to_code", "io_to_code"]


def tiny_config() -> SamplerConfig:
    """Short, single-function programs: arithmetic + lists + control flow, no
    records/variants/match/strings/HOF (keeps sequences short and the signal
    about core syntax/semantics learnability clean)."""
    return SamplerConfig(
        use_bool=True, use_lists=True, use_records=False, use_variants=False,
        use_match=False, use_cond=True, use_foreach=True, use_if=True,
        use_strings=False, use_higher_order=False, use_nested_lists=False,
        min_helpers=0, max_helpers=0, min_params=1, max_params=2,
        max_block_stmts=2, max_loop_stmts=1, block_depth=1,
        expr_fuel=2, helper_fuel=1, max_list_lit=2, int_min=0, int_max=9,
    )


# --------------------------------------------------------------------------
def cmd_prepare(args) -> None:
    os.makedirs(WORK, exist_ok=True)
    print(f"[prepare] generating {args.n} compact programs -> {RAW}")
    t0 = time.time()
    generate(GenSpec(
        out_dir=RAW, n_examples=args.n, base_seed=args.seed,
        use_curriculum=False, sampler_config=tiny_config(),
        n_inputs=4, want_trace=False, want_repair=False, want_refactor=False,
        want_completion=False,
    ))
    stats = export_dataset(RAW, DATA, tasks=TASKS, test_pct=15)
    print(f"[prepare] train={stats['train']} test={stats['test']} "
          f"({time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
def _load(tok_only=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok_only:
        return tok, None
    # eager attention: the optimized/cached GQA attention path miscomputes shapes
    # on MPS during generation (Qwen3 has 16 query / 8 KV heads). Eager is correct.
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("LUCID_DTYPE", "fp32")]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dt, attn_implementation="eager")
    return tok, model


def _read(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _free_mps():
    import gc
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()


def cmd_train(args) -> None:
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(args.seed)
    tok, model = _load()
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.to(DEVICE)
    model.train()
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] device={DEVICE} trainable params={trainable/1e6:.2f}M")

    # Build masked (input_ids, labels) examples; drop any over MAX_LEN.
    eos = tok.eos_token_id
    examples = []
    for ex in _read(os.path.join(DATA, "train.jsonl")):
        p_ids = tok.encode(ex["prompt"])
        c_ids = tok.encode(ex["completion"]) + [eos]
        if len(p_ids) + len(c_ids) > MAX_LEN:
            continue
        ids = p_ids + c_ids
        labels = [-100] * len(p_ids) + c_ids
        examples.append((ids, labels))
    print(f"[train] usable examples (<= {MAX_LEN} tok): {len(examples)}")

    import random
    rng = random.Random(args.seed)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    def make_batch(batch):
        maxlen = max(len(ids) for ids, _ in batch)
        input_ids, attn, labels = [], [], []
        for ids, lab in batch:
            pad = maxlen - len(ids)
            input_ids.append(ids + [tok.pad_token_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [-100] * pad)
        return (torch.tensor(input_ids, device=DEVICE),
                torch.tensor(attn, device=DEVICE),
                torch.tensor(labels, device=DEVICE))

    step = 0
    t0 = time.time()
    accum = args.grad_accum
    opt.zero_grad()
    running = 0.0
    while step < args.steps:
        rng.shuffle(examples)
        for i in range(0, len(examples) - args.batch + 1, args.batch):
            ii, am, lb = make_batch(examples[i:i + args.batch])
            out = model(input_ids=ii, attention_mask=am, labels=lb)
            loss = out.loss / accum
            loss.backward()
            running += out.loss.item()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step()
                opt.zero_grad()
            step += 1
            if step % args.log_every == 0:
                print(f"[train] step {step}/{args.steps} loss={running/args.log_every:.3f} "
                      f"({(time.time()-t0)/step:.2f}s/step)")
                running = 0.0
            if step >= args.steps:
                break
    os.makedirs(ADAPTER, exist_ok=True)
    model.save_pretrained(ADAPTER)
    with open(os.path.join(ADAPTER, "train_config.json"), "w") as f:
        json.dump({
            "steps": args.steps, "batch": args.batch, "grad_accum": args.grad_accum,
            "lr": args.lr, "lora_r": args.lora_r, "dtype": os.environ.get("LUCID_DTYPE", "fp32"),
            "max_len": MAX_LEN, "trainable_params_m": round(trainable / 1e6, 2),
            "usable_examples": len(examples),
        }, f)
    print(f"[train] saved adapter -> {ADAPTER} ({time.time()-t0:.0f}s total)")


# --------------------------------------------------------------------------
@torch.no_grad()
def _generate(tok, model, prompts, max_new=140, batch=8):
    tok.padding_side = "left"
    outs = []
    model.eval()
    n = len(prompts)
    for i in range(0, n, batch):
        chunk = prompts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True).to(DEVICE)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
        print(f"    generated {min(i+batch, n)}/{n}", flush=True)
    return outs


def _eval_model(tok, model, label, limit):
    report = {}
    for task in TASKS:
        items = [t for t in _read(os.path.join(DATA, "test.jsonl")) if t["task"] == task][:limit]
        if not items:
            continue
        gens = _generate(tok, model, [t["prompt"] for t in items])
        res = evaluate(list(zip(items, gens)))
        report[task] = res.summary()
        print(f"[eval:{label}] {task}: {res.summary()}")
    return report


def cmd_eval(args) -> None:
    tok, model = _load()
    if args.adapter and os.path.isdir(args.adapter):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        label = "tuned"
    else:
        label = "base"
    model.to(DEVICE)
    rep = _eval_model(tok, model, label, args.limit)
    print(json.dumps({label: rep}, indent=2))


def cmd_run(args) -> None:
    cmd_prepare(args)
    # base eval
    tok, base = _load()
    base.to(DEVICE)
    base_rep = _eval_model(tok, base, "base", args.limit)
    del base
    _free_mps()
    # train
    cmd_train(args)
    _free_mps()
    # tuned eval
    from peft import PeftModel
    tok2, m = _load()
    m = PeftModel.from_pretrained(m, ADAPTER).to(DEVICE)
    tuned_rep = _eval_model(tok2, m, "tuned", args.limit)
    _write_results(args, base_rep, tuned_rep)


def cmd_report(args) -> None:
    """Evaluate base vs the already-trained adapter and write RESULTS.md
    (no retraining)."""
    tok, base = _load()
    base.to(DEVICE)
    base_rep = _eval_model(tok, base, "base", args.limit)
    del base
    _free_mps()
    from peft import PeftModel
    tok2, m = _load()
    m = PeftModel.from_pretrained(m, ADAPTER).to(DEVICE)
    tuned_rep = _eval_model(tok2, m, "tuned", args.limit)
    _write_results(args, base_rep, tuned_rep)


def _write_results(args, base_rep, tuned_rep) -> None:
    # Prefer the actual training config saved alongside the adapter (the eval/
    # report command's own args are NOT the training hyperparameters).
    tc = {}
    cfg_path = os.path.join(ADAPTER, "train_config.json")
    if os.path.isfile(cfg_path):
        tc = json.load(open(cfg_path))
    ft = (f"LoRA r={tc.get('lora_r', args.lora_r)}, {tc.get('steps', args.steps)} steps, "
          f"batch {tc.get('batch', args.batch)}×{tc.get('grad_accum', args.grad_accum)} accum, "
          f"lr {tc.get('lr', args.lr)}, dtype {tc.get('dtype', '?')}, "
          f"max_len {tc.get('max_len', MAX_LEN)}, "
          f"{tc.get('trainable_params_m', '?')}M trainable on {tc.get('usable_examples', '?')} examples")
    lines = [
        "# Experiment Results — can a small model learn Lucid?",
        "",
        f"- Model: `{MODEL_ID}` (~0.6B params), device `{DEVICE}`",
        f"- Fine-tune: {ft}",
        f"- Data: compact single-function Lucid programs; tasks {TASKS}",
        f"- Eval: greedy decoding, up to {args.limit} held-out items per task",
        "",
        "Metrics (PRD §4): parse-rate and typecheck-rate are H1 (well-formedness);",
        "exec_pass@1 is H2 (runs and reproduces the held-out IO). exact_match is a",
        "bonus (canonical-identical to the reference).",
        "",
    ]
    for task in TASKS:
        b = base_rep.get(task, {})
        t = tuned_rep.get(task, {})
        lines += [
            f"## Task: `{task}`",
            "",
            "| metric | base | fine-tuned |",
            "|---|---|---|",
        ]
        for k in ["parse_rate", "typecheck_rate", "exec_pass@1", "exact_match"]:
            lines.append(f"| {k} | {b.get(k, '-')} | {t.get(k, '-')} |")
        lines.append("")
    lines += [
        "## Reading the result",
        "",
        "H1 holds if fine-tuned typecheck-rate ≫ base. H2 holds if fine-tuned",
        "exec_pass@1 ≫ base. Because Lucid is canonical, total, and executable,",
        "these are measured exactly by the toolchain — no human grading.",
        "",
        f"_Raw: base={json.dumps(base_rep)}  tuned={json.dumps(tuned_rep)}_",
    ]
    out = os.path.join(ROOT, "RESULTS.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[run] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["prepare", "train", "eval", "run", "report"]:
        p = sub.add_parser(name)
        p.add_argument("--n", type=int, default=6000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--steps", type=int, default=400)
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--grad-accum", type=int, default=2, dest="grad_accum")
        p.add_argument("--lr", type=float, default=2e-4)
        p.add_argument("--lora-r", type=int, default=16, dest="lora_r")
        p.add_argument("--log-every", type=int, default=25, dest="log_every")
        p.add_argument("--limit", type=int, default=80)
        p.add_argument("--adapter", type=str, default=ADAPTER)
        p.set_defaults(func={"prepare": cmd_prepare, "train": cmd_train,
                             "eval": cmd_eval, "run": cmd_run,
                             "report": cmd_report}[name])
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
