#!/usr/bin/env bash
#SBATCH --job-name=llmtd-subgraph-adapt
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="${ANALYSIS_RUN_ROOT:-$A/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
OUT="$R/subgroup_amount_adapter_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/mixed_type_condition_adapter.py" \
  --repo "$REPO" --dataset default \
  --config "$R/full_train_default_1033760/config.pkl" \
  --checkpoint "$R/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --output "$OUT" --representation structured \
  --epochs 60 --pairs-per-contract 768 --batch-size 2048 --eval-samples 512 \
  --guidance-scales 1:2 --adapter-lr 5e-4 --core-lr 1e-5 \
  --condition-dropout 0.1 --threshold-curriculum fixed \
  --run-label subgroup_amount_atomic --seed 20260925
