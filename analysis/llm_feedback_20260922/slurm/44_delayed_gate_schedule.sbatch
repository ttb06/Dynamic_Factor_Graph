#!/usr/bin/env bash
#SBATCH --job-name=llmtd-gatesched
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${SCHED_ADAPTER:?missing adapter path}"
: "${SCHED_SEED:?missing seed}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
EXTRA=()
if [[ "${SCHED_GLOBAL_EARLY:-0}" == "1" ]]; then
  EXTRA+=(--include-global-early)
fi
cd "$REPO"
"$PY" "$ANALYSIS/scripts/delayed_gate_schedule_probe.py" \
  --repo "$REPO" \
  --config "$RUN_ROOT/full_train_1028011/config.pkl" \
  --checkpoint "$RUN_ROOT/full_train_1028011/ema_model_epoch_8000.pt" \
  --adapter "$SCHED_ADAPTER" \
  --output "$RUN_ROOT/delayed_gate_schedule_${SCHED_SEED}_${SLURM_JOB_ID}" \
  --samples "${SCHED_SAMPLES:-1024}" \
  --steps "${SCHED_STEPS:-30}" \
  --guidance "${SCHED_GUIDANCE:-2}" \
  --gate-starts "${SCHED_STARTS:-1:0.5:0.25:0.1:-1}" \
  --seed "$SCHED_SEED" \
  "${EXTRA[@]}"
