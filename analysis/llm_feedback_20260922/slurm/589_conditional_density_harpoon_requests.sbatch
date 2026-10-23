#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cond-harpoon
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
requests=(range category)
i=$SLURM_ARRAY_TASK_ID; request=${requests[$i]}

cd "$REPO"
"$PY" "$A/scripts/conditional_density_graph_repaint.py" \
  --repo "$REPO" --dataset default \
  --graph "$A/outputs/harpoon_default_graphs_1065641/${request}.json" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --checkpoint-time .25 --max-calibration-rate .05 \
  --branches 8 --rounds 2 --density-weight 1.0 --conditional-weight 1.0 \
  --samples 4096 --steps 50 --seed 20261376 \
  --output "$A/outputs/conditional_density_harpoon_${request}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
