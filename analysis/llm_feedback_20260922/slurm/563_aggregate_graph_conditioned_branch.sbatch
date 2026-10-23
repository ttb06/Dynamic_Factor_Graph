#!/usr/bin/env bash
#SBATCH --job-name=llmtd-gca-agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${ONE_JOB:?}"; : "${ROLL_JOB:?}"; : "${BSCG_JOB:?}"; : "${JOINT_JOB:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$A/scripts/aggregate_graph_conditioned_branch.py" --outputs "$A/outputs" \
  --experiment "one_step=$ONE_JOB" --experiment "trajectory_unrolled=$ROLL_JOB" \
  --experiment "bootstrap_graph_unrolled=$BSCG_JOB" --experiment "joint_end_to_end=$JOINT_JOB" \
  --output "$A/outputs/graph_conditioned_branch_aggregate_${SLURM_JOB_ID}"
