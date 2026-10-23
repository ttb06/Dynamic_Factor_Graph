#!/usr/bin/env bash
#SBATCH --job-name=llmtd-msg-grid
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
labels=(weak medium strong)
etas=(2 4 8)
duals=(1 2 4)
label=${labels[$SLURM_ARRAY_TASK_ID]}
cd "$REPO"
"$PY" "$A/scripts/dynamic_graph_logit_messages.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261376 --start-t .50 \
  --eta "${etas[$SLURM_ARRAY_TASK_ID]}" --dual-lr "${duals[$SLURM_ARRAY_TASK_ID]}" \
  --output "$A/outputs/dynamic_messages_${label}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
