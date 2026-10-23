#!/usr/bin/env bash
#SBATCH --job-name=llmtd-nollm-splitmsg
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
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SPLIT_JOB_ID=${SPLIT_JOB_ID:?SPLIT_JOB_ID required}
split_seeds=(20261006 20261007)
split_seed=${split_seeds[$SLURM_ARRAY_TASK_ID]}
GRAPH="$A/outputs/no_llm_graph_split_s${split_seed}_${SPLIT_JOB_ID}_${SLURM_ARRAY_TASK_ID}/graph.json"
cd "$REPO"
"$PY" "$A/scripts/dynamic_graph_logit_messages.py" --repo "$REPO" --dataset default \
  --graph "$GRAPH" --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261376 --start-t .50 --eta 8 --dual-lr 4 \
  --output "$A/outputs/no_llm_splitgraph_messages_s${split_seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
