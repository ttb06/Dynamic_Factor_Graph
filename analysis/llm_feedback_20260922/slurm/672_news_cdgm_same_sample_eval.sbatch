#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-cdgm-same
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${SOURCE_RUN_ID:?SOURCE_RUN_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SOURCE="$A/outputs/news_projection_balanced_l2_${SOURCE_RUN_ID}/samples"
OUT="$A/outputs/news_cdgm_same_sample_eval_${SLURM_JOB_ID}"
ARTIFACT="$A/outputs/news_cdgm_same_sample_artifacts_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/apply_constrained_dgm_news_layer.py" \
  --cdgm-repo "$ROOT/baseline_repos/constrained_dgm" \
  --sample "$SOURCE/unconditional.csv" \
  --output "$ARTIFACT/cdgm_constraint_layer.csv"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$SOURCE/unconditional.csv" \
  --sample "final_repair=$SOURCE/final_reward_calibrated.csv" \
  --sample "dynamic_online=$SOURCE/x0_reward_calibrated.csv" \
  --sample "dynamic_then_certifier=$SOURCE/x0_reward_calibrated_then_final_reward.csv" \
  --sample "iclr24_constraint_layer=$ARTIFACT/cdgm_constraint_layer.csv" \
  --seed 20261447 --output "$OUT"
