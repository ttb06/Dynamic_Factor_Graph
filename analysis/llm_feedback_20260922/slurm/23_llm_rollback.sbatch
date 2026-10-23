#!/usr/bin/env bash
#SBATCH --job-name=llmtd-rollback
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${ROLLBACK_CONDITION:?missing condition}"
: "${ROLLBACK_TAG:?missing tag}"
: "${ROLLBACK_T:?missing timestep}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/llm_rollback_sampler.py" \
  --repo "$REPO" \
  --checkpoint "$ANALYSIS/outputs/full_train_1028011/ema_model_epoch_8000.pt" \
  --config "$ANALYSIS/outputs/full_train_1028011/config.pkl" \
  --env-file "$ROOT/EIPC/.env" \
  --output "$RUN_ROOT/llm_rollback_${ROLLBACK_TAG}_${SLURM_JOB_ID}" \
  --condition "$ROLLBACK_CONDITION" \
  --model "${ROLLBACK_MODEL:-gpt-4.1-mini}" \
  --target-t "$ROLLBACK_T" \
  --samples 96 \
  --steps 30 \
  --seed 20260941 \
  --batch-size 12 \
  --inter-call-delay 4
