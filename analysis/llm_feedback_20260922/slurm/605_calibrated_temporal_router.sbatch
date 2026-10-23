#!/usr/bin/env bash
#SBATCH --job-name=llmtd-time-router
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-5
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(shoppers shoppers shoppers default default default)
seeds=(20261444 20261445 20261446 20261444 20261445 20261446)
source_runs=(full_train_shoppers_1033761 full_train_shoppers_1033761 full_train_shoppers_1033761 full_train_default_1033760 full_train_default_1033760 full_train_default_1033760)
graph_indices=(1 1 1 0 0 0)
calibration_runs=(trajectory_repaint_shoppers_union_1070867_2 trajectory_repaint_shoppers_union_1070867_2 trajectory_repaint_shoppers_union_1070867_2 trajectory_repaint_default_union_1070867_5 trajectory_repaint_default_union_1070867_5 trajectory_repaint_default_union_1070867_5)
i=$SLURM_ARRAY_TASK_ID; dataset=${datasets[$i]}; seed=${seeds[$i]}; source_run=${source_runs[$i]}
graph_index=${graph_indices[$i]}; calibration_run=${calibration_runs[$i]}
cd "$REPO"
"$PY" "$A/scripts/calibrated_temporal_router_repaint.py" \
  --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/bootstrap_conditional_graph_${dataset}_1070443_${graph_index}/graph.json" \
  --calibration-dir "$A/outputs/$calibration_run/samples" \
  --config "$A/outputs/$source_run/config.pkl" --checkpoint "$A/outputs/$source_run/ema_model_epoch_8000.pt" \
  --checkpoints .40,.25 --max-calibration-rate .05 --branches 8 --rounds 2 --density-weight 1.0 \
  --samples 4096 --steps 50 --seed "$seed" \
  --output "$A/outputs/calibrated_temporal_router_${dataset}_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
