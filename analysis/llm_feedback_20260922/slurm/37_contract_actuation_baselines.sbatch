#!/usr/bin/env bash
#SBATCH --job-name=llmtd-actbase
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${ACT_SEED:?missing seed}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/contract_actuation_baselines.py" \
  --repo "$REPO" \
  --config "$RUN_ROOT/full_train_1028011/config.pkl" \
  --checkpoint "$RUN_ROOT/full_train_1028011/ema_model_epoch_8000.pt" \
  --output "$RUN_ROOT/contract_actuation_${ACT_SEED}_${SLURM_JOB_ID}" \
  --samples "${ACT_SAMPLES:-256}" \
  --steps "${ACT_STEPS:-30}" \
  --best-of "${ACT_BEST_OF:-2,4}" \
  --seed "$ACT_SEED"
