#!/usr/bin/env bash
#SBATCH --job-name=llmtd-row-timeline
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/export_row_edge_timeline.py" \
  --repo "$REPO" --dataset default \
  --graph "$A/outputs/bootstrap_conditional_graph_default_1070443_0/graph.json" \
  --schedule "$A/outputs/bscg_dynamic_trajectory_default_1080086_0/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261440 --examples 6 \
  --output "$A/outputs/row_edge_timeline_default_${SLURM_JOB_ID}"
