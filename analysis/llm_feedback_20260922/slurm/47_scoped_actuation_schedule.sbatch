#!/bin/bash
#SBATCH --job-name=scoped-act
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${SCOPE_ADAPTER:?missing adapter path}"
: "${SCOPE_GUIDANCE:?missing guidance scale}"
: "${SCOPE_SEED:?missing seed}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
ANALYSIS="$ROOT/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
OUT="$RUN_ROOT/scoped_actuation_g${SCOPE_GUIDANCE}_seed${SCOPE_SEED}_${SLURM_JOB_ID}"

cd "$ROOT"
"$PY" "$ANALYSIS/scripts/scoped_actuation_schedule_probe.py" \
  --repo "$ROOT" \
  --config "$RUN_ROOT/full_train_1028011/config.pkl" \
  --checkpoint "$RUN_ROOT/full_train_1028011/ema_model_epoch_8000.pt" \
  --adapter "$SCOPE_ADAPTER" \
  --output "$OUT" --samples 1024 --steps 30 \
  --guidance "$SCOPE_GUIDANCE" --seed "$SCOPE_SEED"
