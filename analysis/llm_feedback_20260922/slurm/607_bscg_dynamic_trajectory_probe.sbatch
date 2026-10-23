#!/usr/bin/env bash
#SBATCH --job-name=llmtd-bscg-dynprobe
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(default shoppers); graph_indices=(0 1)
source_runs=(full_train_default_1033760 full_train_shoppers_1033761)
i=$SLURM_ARRAY_TASK_ID; dataset=${datasets[$i]}; graph_index=${graph_indices[$i]}; source_run=${source_runs[$i]}
cd "$REPO"
"$PY" "$A/scripts/typed_graph_trajectory_probe.py" \
  --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/bootstrap_conditional_graph_${dataset}_1070443_${graph_index}/graph.json" \
  --config "$A/outputs/$source_run/config.pkl" --checkpoint "$A/outputs/$source_run/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261440 --precision .90 --recall .80 \
  --output "$A/outputs/bscg_dynamic_trajectory_${dataset}_${SLURM_ARRAY_JOB_ID}_${i}"
