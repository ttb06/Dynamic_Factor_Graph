#!/bin/bash
#SBATCH --job-name=llmtd-graph-gate
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
cd "$REPO"
OUT="$REPO/analysis/llm_feedback_20260922/outputs/persistence_graph_gate_${SLURM_JOB_ID}"
/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python analysis/llm_feedback_20260922/scripts/subgroup_trajectory_probe.py \
  --repo "$REPO" --dataset default \
  --audit "$REPO/analysis/llm_feedback_20260922/outputs/subgroup_graph_audit_1050323" \
  --config "$REPO/analysis/llm_feedback_20260922/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$REPO/analysis/llm_feedback_20260922/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --output "$OUT" --samples 2048 --steps 50 --max-candidates 12 --seed 20261001
