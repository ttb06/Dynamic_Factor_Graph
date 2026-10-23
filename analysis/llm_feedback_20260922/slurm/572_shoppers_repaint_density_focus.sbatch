#!/usr/bin/env bash
#SBATCH --job-name=llmtd-repaint-density
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-4
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"; PY="$ROOT/baseline_runs/envs/grdm/bin/python"
labels=(density15 density20 b12 margin25 rarity25)
density=(1.5 2.0 1.0 1.0 1.0); branches=(8 8 12 8 8); margins=(0 0 0 .25 0); rarity=(.1 .1 .1 .1 .25)
i=$SLURM_ARRAY_TASK_ID; label=${labels[$i]}
cd "$REPO"
"$PY" "$A/scripts/dynamic_graph_conditional_repaint.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/bootstrap_conditional_graph_shoppers_1070443_1/graph.json" \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --trigger mid --checkpoint-time .25 --max-calibration-rate .05 --rounds 2 \
  --branches "${branches[$i]}" --density-weight "${density[$i]}" \
  --rarity-weight "${rarity[$i]}" --accept-margin "${margins[$i]}" \
  --samples 2048 --steps 50 --seed 20261440 \
  --output "$A/outputs/dg_repaint_density_shoppers_${label}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
