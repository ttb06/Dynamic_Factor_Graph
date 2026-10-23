#!/usr/bin/env bash
#SBATCH --job-name=llmtd-bscg-roll-s
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-3
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${GRAPH_JOB_ID:?GRAPH_JOB_ID required}"; : "${TRAIN_JOB_ID:?TRAIN_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(shoppers shoppers diabetes diabetes); variants=(topology no_topology topology no_topology); graph_indices=(1 1 2 2)
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}; variant=${variants[$SLURM_ARRAY_TASK_ID]}; gi=${graph_indices[$SLURM_ARRAY_TASK_ID]}
if [[ "$dataset" == shoppers ]]; then source_run=full_train_shoppers_1033761
else source_run=full_train_diabetes_1048479; fi
cd "$REPO"
"$PY" "$A/scripts/graph_conditioned_residual_denoiser.py" sample --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/bootstrap_conditional_graph_${dataset}_${GRAPH_JOB_ID}_${gi}/graph.json" \
  --config "$A/outputs/$source_run/config.pkl" --checkpoint "$A/outputs/$source_run/ema_model_epoch_8000.pt" \
  --adapter "$A/outputs/bscg_adapter_${dataset}_${variant}_${TRAIN_JOB_ID}_${SLURM_ARRAY_TASK_ID}/adapter_epoch_80.pt" \
  --samples 4096 --steps 50 --seed 20261430 \
  --output "$A/outputs/bscg_adapter_sample_${dataset}_${variant}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
