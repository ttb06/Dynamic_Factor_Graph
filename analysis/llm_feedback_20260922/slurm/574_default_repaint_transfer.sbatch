#!/usr/bin/env bash
#SBATCH --job-name=llmtd-repaint-transfer
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
labels=(density1 density2); density=(1.0 2.0); i=$SLURM_ARRAY_TASK_ID; label=${labels[$i]}
cd "$REPO"
"$PY" "$A/scripts/dynamic_graph_conditional_repaint.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/bootstrap_conditional_graph_default_1070443_0/graph.json" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --trigger mid --checkpoint-time .25 --max-calibration-rate .05 --rounds 2 \
  --branches 8 --density-weight "${density[$i]}" --samples 2048 --steps 50 --seed 20261440 \
  --output "$A/outputs/dg_repaint_transfer_default_${label}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
