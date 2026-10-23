#!/usr/bin/env bash
#SBATCH --job-name=llmtd-policygrid-D
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="$A/outputs"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
OUT="$R/default_policy_difficulty_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/subgroup_conditioned_sampling.py" --repo "$REPO" --dataset default \
  --config "$R/full_train_default_1033760/config.pkl" --checkpoint "$R/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --adapter "$R/subgroup_amount_adapter_1050333/adapter.pt" --output "$OUT" \
  --samples 1536 --steps 30 --seed 20260925 --guidances 2 --policy-grid --focused
