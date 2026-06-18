#!/usr/bin/env bash
# Matched Lucid-vs-Python H1-H3 sweep (fair: cap on USABLE <=max_len examples).
#
# Memory-safe config for a 16GB Apple Silicon Mac (see PROGRESS.md "Cycle 2"):
#   * bf16            -> 2x less memory than fp32 (fp32 swap-thrashed: 40-74s/step)
#   * batch 2 / accum 2 -> half the activation peak, same effective batch (4)
#   * max_len 384     -> keeps >=99% of BOTH languages so the two corpora are the
#                        same underlying programs (matched-program invariant)
#   * compare.py flushes the MPS allocator every 25 steps (stops the swap creep)
#
# Resumable: finished cells append to results.jsonl; reruns skip them.
set -euo pipefail
cd "$(dirname "$0")/.."

export SSL_CERT_FILE="$(python3 -c 'import certifi; print(certifi.where())')"
export LUCID_DTYPE=bf16
export LUCID_MAXLEN=384
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1

SIZES="${SIZES:-500,1000,2000}"
EPOCHS="${EPOCHS:-2}"
LIMIT="${LIMIT:-60}"

LOG="experiment/compare/sweep_fair_v2.log"
echo "started $(date)" | tee "$LOG"
python3 scripts/compare.py run \
  --sizes "$SIZES" --epochs "$EPOCHS" --limit "$LIMIT" \
  --batch 2 --grad-accum 2 --lr 3e-4 --lora-r 16 2>&1 | tee -a "$LOG"
echo "finished $(date)" | tee -a "$LOG"
