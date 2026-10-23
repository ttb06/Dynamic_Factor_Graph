#!/usr/bin/env bash
#SBATCH --job-name=llmtd-subgraph-traj
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="${ANALYSIS_RUN_ROOT:-$A/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
OUT="$R/subgroup_trajectory_default_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/subgroup_trajectory_probe.py" \
  --repo "$REPO" --dataset default \
  --audit "$R/subgroup_graph_audit_1050323" \
  --config "$R/full_train_default_1033760/config.pkl" \
  --checkpoint "$R/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --output "$OUT" --samples 1024 --steps 30 --max-candidates 8 --seed 20260925
