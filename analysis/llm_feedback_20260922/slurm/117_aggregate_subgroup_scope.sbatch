#!/usr/bin/env bash
#SBATCH --job-name=llmtd-subgraph-scopeagg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
R="$A/outputs"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
mapfile -t CONFIRM < <(find "$R" -maxdepth 1 -type d -name 'subgroup_scope_confirm_seed*_1050686_*' | sort)
if [[ ${#CONFIRM[@]} -ne 2 ]]; then
  echo "expected two scope-confirmation runs, got ${#CONFIRM[@]}" >&2
  exit 2
fi
OUT="$R/subgroup_scope_aggregate_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/aggregate_subgroup_confirm.py" \
  --runs "$R/subgroup_actuation_scope_1050684" "${CONFIRM[@]}" \
  --output "$OUT" --bootstrap 10000 --seed 20260930
