#!/usr/bin/env bash
#SBATCH --job-name=llmtd-gca-sample
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-7
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${TRAIN_JOB_ID:?TRAIN_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
datasets=(shoppers shoppers shoppers shoppers diabetes diabetes diabetes diabetes)
variants=(balanced balanced no_topology graph_strong balanced balanced no_topology graph_strong)
seeds=(20261420 20261421 20261420 20261420 20261420 20261421 20261420 20261420)
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}; variant=${variants[$SLURM_ARRAY_TASK_ID]}; seed=${seeds[$SLURM_ARRAY_TASK_ID]}
if [[ "$dataset" == shoppers ]]; then
  source_run=full_train_shoppers_1033761; graph=actionable_graph_v2_shoppers_1060952_0/empirical_graph.json
else
  source_run=full_train_diabetes_1048479; graph=actionable_graph_v2_diabetes_1060952_2/empirical_graph.json
fi
train_dir="$A/outputs/graph_adapter_${dataset}_${variant}_s${seed}_${TRAIN_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/graph_conditioned_residual_denoiser.py" sample \
  --repo "$REPO" --dataset "$dataset" --graph "$A/outputs/$graph" \
  --config "$A/outputs/$source_run/config.pkl" --checkpoint "$A/outputs/$source_run/ema_model_epoch_8000.pt" \
  --adapter "$train_dir/adapter_epoch_100.pt" --samples 4096 --steps 50 --seed "$seed" \
  --output "$A/outputs/graph_adapter_sample_${dataset}_${variant}_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
