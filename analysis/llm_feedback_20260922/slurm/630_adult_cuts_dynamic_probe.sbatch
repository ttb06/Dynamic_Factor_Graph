#!/usr/bin/env bash
#SBATCH --job-name=llmtd-adult-cuts-p
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${TRAIN_JOB_ID:?TRAIN_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
GRAPH="$A/outputs/adult_cuts_typed_graph_v2_20261015/graph.json"
cd "$REPO"
"$PY" "$A/scripts/typed_graph_trajectory_probe.py" --repo "$REPO" --dataset adult \
  --graph "$GRAPH" \
  --config "$A/outputs/extended_full_train_adult_${TRAIN_JOB_ID}/config.pkl" \
  --checkpoint "$A/outputs/extended_full_train_adult_${TRAIN_JOB_ID}/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261440 --precision .90 --recall .80 \
  --output "$A/outputs/adult_cuts_dynamic_trajectory_${SLURM_JOB_ID}"
