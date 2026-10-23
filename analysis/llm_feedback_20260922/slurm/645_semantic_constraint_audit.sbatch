#!/usr/bin/env bash
#SBATCH --job-name=llmtd-semantic-audit
#SBATCH --partition=cpu
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${DISCOVERY_JOB_ID:?DISCOVERY_JOB_ID required}"

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
profiles=(complex_hyperedge trajectory_actionable)
i=$SLURM_ARRAY_TASK_ID
profile=${profiles[$i]}
run="$A/outputs/default_semantic_discovery_${profile}_${DISCOVERY_JOB_ID}_${i}"
output="$A/outputs/default_semantic_audit_${profile}_${SLURM_ARRAY_JOB_ID}_${i}"

cd "$REPO"
"$PY" "$A/scripts/audit_constraint_meaningfulness.py" \
  --run "$run" --repo "$REPO" --dataset default --output "$output"
