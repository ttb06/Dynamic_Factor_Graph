#!/usr/bin/env bash
#SBATCH --job-name=llmtd-impute-scopeS
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
mapfile -t RUNS < <(find "$R" -maxdepth 1 -type d -name 'matched_impute_shoppers_seed*' | sort)
[[ ${#RUNS[@]} -eq 3 ]] || { echo "expected three Shoppers runs, got ${#RUNS[@]}" >&2; exit 2; }
OUT="$R/matched_impute_shoppers_aggregate_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/aggregate_subgroup_confirm.py" --runs "${RUNS[@]}" \
  --output "$OUT" --bootstrap 10000 --seed 20261004
