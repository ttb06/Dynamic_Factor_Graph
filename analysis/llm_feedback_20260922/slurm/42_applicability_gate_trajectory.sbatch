#!/usr/bin/env bash
#SBATCH --job-name=llmtd-gatetrace
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${GATE_ADAPTER:?missing adapter path}"
: "${GATE_SEED:?missing seed}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/applicability_gate_trajectory.py" \
  --repo "$REPO" \
  --config "$RUN_ROOT/full_train_1028011/config.pkl" \
  --checkpoint "$RUN_ROOT/full_train_1028011/ema_model_epoch_8000.pt" \
  --adapter "$GATE_ADAPTER" \
  --output "$RUN_ROOT/applicability_gate_trajectory_${GATE_SEED}_${SLURM_JOB_ID}" \
  --samples "${GATE_SAMPLES:-512}" \
  --steps "${GATE_STEPS:-30}" \
  --guidance-scales "${GATE_GUIDANCE:-1:2:4}" \
  --seed "$GATE_SEED"
