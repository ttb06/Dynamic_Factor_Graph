#!/usr/bin/env bash
#SBATCH --job-name=llmtd-diab-probe
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/typed_graph_trajectory_probe.py" \
  --repo "$REPO" --dataset diabetes \
  --graph "$A/outputs/bootstrap_conditional_graph_diabetes_1070443_2/graph.json" \
  --config "$A/outputs/full_train_diabetes_1048479/config.pkl" \
  --checkpoint "$A/outputs/full_train_diabetes_1048479/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261440 --precision .90 --recall .80 \
  --output "$A/outputs/bscg_dynamic_trajectory_diabetes_${SLURM_JOB_ID}"
