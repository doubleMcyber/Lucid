"""Head-to-head language thesis experiment: Lucid vs a matched Python baseline.

Tests PRD hypotheses H1 (well-formedness), H2 (executed pass@1), H3 (data
efficiency: accuracy vs #examples), and H4 (small-vs-large model gap) by training
the SAME base model on the SAME task distribution in two surface languages and
comparing. The Python corpus is transpiled from the identical Loom programs, with
the identical train/test split, so only the language differs.

Protocol
  * One shared raw corpus; export Lucid and Python with the same split.
  * A "cell" = (lang, model_id, n_train, epochs). Compute is matched between the
    two languages at each (model_id, n_train) because the corpora are matched
    (same example count) and epochs are equal => equal optimizer steps.
  * Eval: greedy decode `limit` held-out items per task; score with the toolchain
    (Lucid) / sandbox (Python). Headline metric = executed pass@1 on held-out IO.
  * Resumable: each finished cell is appended to results.jsonl; reruns skip it.

Usage:
  LUCID_DTYPE=bf16 LUCID_MAXLEN=192 python scripts/compare.py prepare --n 6000
  LUCID_DTYPE=bf16 LUCID_MAXLEN=192 python scripts/compare.py run --sizes 1500,3000,6000 --epochs 2 --limit 60
  python scripts/compare.py report
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for `experiment`

import experiment as E  # noqa: E402  (sets HF offline env, imports torch)
import torch  # noqa: E402

from loom.config import SamplerConfig  # noqa: E402
from loom.evaluate import evaluate  # noqa: E402
from loom.evaluate_py import evaluate_py  # noqa: E402
from loom.export import export_dataset  # noqa: E402
from loom.export_py import export_python_dataset  # noqa: E402
from loom.leakcheck import audit_split  # noqa: E402
from loom.pipeline import GenSpec, generate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "experiment", "compare")
RAW = os.path.join(WORK, "raw")
LUCID_DIR = os.path.join(WORK, "lucid")
PY_DIR = os.path.join(WORK, "py")
RESULTS = os.path.join(WORK, "results.jsonl")
TASKS = ["spec_to_code", "io_to_code"]
DEVICE = E.DEVICE


def _tiny() -> SamplerConfig:
    return SamplerConfig(
        use_bool=True, use_lists=True, use_records=False, use_variants=False,
        use_match=False, use_cond=True, use_foreach=True, use_if=True,
        use_strings=False, use_higher_order=False, use_nested_lists=False,
        min_helpers=0, max_helpers=0, min_params=1, max_params=2,
        max_block_stmts=2, max_loop_stmts=1, block_depth=1,
        expr_fuel=2, helper_fuel=1, max_list_lit=2, int_min=0, int_max=9,
    )


def cmd_prepare(args) -> None:
    os.makedirs(WORK, exist_ok=True)
    print(f"[prepare] generating {args.n} programs -> {RAW}", flush=True)
    t0 = time.time()
    generate(GenSpec(out_dir=RAW, n_examples=args.n, base_seed=args.seed,
                     use_curriculum=False, sampler_config=_tiny(),
                     n_inputs=4, want_trace=False, want_repair=False,
                     want_refactor=False, want_completion=False))
    ls = export_dataset(RAW, LUCID_DIR, tasks=TASKS, test_pct=args.test_pct)
    ps = export_python_dataset(RAW, PY_DIR, tasks=TASKS, test_pct=args.test_pct)
    # both splits must be leak-free
    for name, d in (("lucid", LUCID_DIR), ("python", PY_DIR)):
        rep = audit_split(os.path.join(d, "train.jsonl"), os.path.join(d, "test.jsonl"))
        assert rep.clean, f"{name} split leaks: {rep.summary()}"
    print(f"[prepare] lucid train/test={ls['train']}/{ls['test']}  "
          f"python={ps['train']}/{ps['test']}  ({time.time()-t0:.1f}s)  both CLEAN", flush=True)


# --------------------------------------------------------------------------
def _build_examples(tok, path: str, n_cap: int, seed: int):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n_cap]
    eos = tok.eos_token_id
    examples = []
    for ex in rows:
        p_ids = tok.encode(ex["prompt"])
        c_ids = tok.encode(ex["completion"]) + [eos]
        if len(p_ids) + len(c_ids) > E.MAX_LEN:
            continue
        examples.append((p_ids + c_ids, [-100] * len(p_ids) + c_ids))
    return examples


def _train(tok, model, examples, epochs, batch, grad_accum, lr, lora_r, seed):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    lcfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.to(DEVICE)
    model.train()
    model.config.use_cache = False
    steps = max(1, epochs * (len(examples) // batch))

    def make_batch(batch_rows):
        maxlen = max(len(ids) for ids, _ in batch_rows)
        ii, am, lb = [], [], []
        for ids, lab in batch_rows:
            pad = maxlen - len(ids)
            ii.append(ids + [tok.pad_token_id] * pad)
            am.append([1] * len(ids) + [0] * pad)
            lb.append(lab + [-100] * pad)
        return (torch.tensor(ii, device=DEVICE), torch.tensor(am, device=DEVICE),
                torch.tensor(lb, device=DEVICE))

    rng = random.Random(seed)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    step = 0
    t0 = time.time()
    opt.zero_grad()
    running = 0.0
    while step < steps:
        rng.shuffle(examples)
        for i in range(0, len(examples) - batch + 1, batch):
            ii, am, lb = make_batch(examples[i:i + batch])
            out = model(input_ids=ii, attention_mask=am, labels=lb)
            (out.loss / grad_accum).backward()
            running += out.loss.item()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step()
                opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"      step {step}/{steps} loss={running/100:.3f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
                running = 0.0
            if step >= steps:
                break
    return model, steps


def _eval(tok, model, lang, test_path, limit):
    rep = {}
    for task in TASKS:
        items = [json.loads(l) for l in open(test_path) if l.strip()]
        items = [t for t in items if t["task"] == task][:limit]
        if not items:
            continue
        gens = E._generate(tok, model, [t["prompt"] for t in items])
        if lang == "lucid":
            rep[task] = evaluate(list(zip(items, gens))).summary()
        else:
            rep[task] = evaluate_py(list(zip(items, gens))).summary()
        print(f"      [{lang}:{task}] {rep[task]}", flush=True)
    return rep


def _cell_key(lang, model_id, n_train, epochs):
    return f"{lang}|{model_id}|{n_train}|{epochs}"


def _done_cells():
    if not os.path.isfile(RESULTS):
        return set()
    return {json.loads(l)["key"] for l in open(RESULTS) if l.strip()}


def _run_cell(lang, model_id, n_train, epochs, args):
    key = _cell_key(lang, model_id, n_train, epochs)
    test_path = os.path.join(LUCID_DIR if lang == "lucid" else PY_DIR, "test.jsonl")
    train_path = os.path.join(LUCID_DIR if lang == "lucid" else PY_DIR, "train.jsonl")
    print(f"[cell] {key}", flush=True)
    E.MODEL_ID = model_id
    tok, base = E._load()
    examples = _build_examples(tok, train_path, n_train, args.seed)
    base.to(DEVICE)
    model, steps = _train(tok, base, examples, epochs, args.batch,
                          args.grad_accum, args.lr, args.lora_r, args.seed)
    rep = _eval(tok, model, lang, test_path, args.limit)
    rec = {"key": key, "lang": lang, "model_id": model_id, "n_train": n_train,
           "epochs": epochs, "steps": steps, "usable_examples": len(examples),
           "report": rep}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    del base, model
    E._free_mps()
    return rec


def cmd_run(args) -> None:
    sizes = [int(s) for s in args.sizes.split(",")]
    models = [m for m in args.models.split(",") if m]
    done = _done_cells()
    plan = []
    for model_id in models:
        for n in sizes:
            for lang in ("lucid", "python"):
                k = _cell_key(lang, model_id, n, args.epochs)
                if k not in done:
                    plan.append((lang, model_id, n, args.epochs))
    print(f"[run] {len(plan)} cells to run ({len(done)} already done)", flush=True)
    for (lang, model_id, n, ep) in plan:
        _run_cell(lang, model_id, n, ep, args)
    cmd_report(args)


def cmd_report(args) -> None:
    if not os.path.isfile(RESULTS):
        print("no results yet")
        return
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]

    def execpass(r):
        rep = r["report"]
        vals = [rep[t]["exec_pass@1"] for t in TASKS if t in rep]
        return sum(vals) / len(vals) if vals else 0.0

    lines = ["# Lucid vs Python — matched-baseline results", "",
             "Executed pass@1 (mean over tasks), held-out IO. Same base model, same",
             "task distribution, matched data + epochs per (model, n_train).", ""]
    models = sorted({r["model_id"] for r in rows})
    sizes = sorted({r["n_train"] for r in rows})
    for model_id in models:
        lines += [f"## {model_id}", "",
                  "| n_train | Lucid exec@1 | Python exec@1 | Δ (L−P) |",
                  "|---|---|---|---|"]
        for n in sizes:
            L = next((r for r in rows if r["model_id"] == model_id and r["n_train"] == n and r["lang"] == "lucid"), None)
            P = next((r for r in rows if r["model_id"] == model_id and r["n_train"] == n and r["lang"] == "python"), None)
            if not L and not P:
                continue
            lv = execpass(L) if L else None
            pv = execpass(P) if P else None
            d = f"{lv-pv:+.3f}" if (lv is not None and pv is not None) else "-"
            lines.append(f"| {n} | {lv if lv is not None else '-'} | {pv if pv is not None else '-'} | {d} |")
        lines.append("")
    # per-task detail
    lines += ["## Per-task detail", "",
              "| cell | task | parse | type/runs | exec@1 |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: x["key"]):
        for t in TASKS:
            if t in r["report"]:
                s = r["report"][t]
                second = s.get("typecheck_rate", s.get("runs_rate", "-"))
                lines.append(f"| {r['key']} | {t} | {s['parse_rate']} | {second} | {s['exec_pass@1']} |")
    out = os.path.join(ROOT, "BASELINE.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[report] wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["prepare", "run", "report"]:
        p = sub.add_parser(name)
        p.add_argument("--n", type=int, default=6000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--test-pct", type=int, default=15, dest="test_pct")
        p.add_argument("--sizes", type=str, default="1500,3000,6000")
        p.add_argument("--models", type=str, default="Qwen/Qwen3-0.6B-Base")
        p.add_argument("--epochs", type=int, default=2)
        p.add_argument("--batch", type=int, default=4)
        p.add_argument("--grad-accum", type=int, default=1, dest="grad_accum")
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--lora-r", type=int, default=16, dest="lora_r")
        p.add_argument("--limit", type=int, default=60)
        p.set_defaults(func={"prepare": cmd_prepare, "run": cmd_run, "report": cmd_report}[name])
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
