#!/usr/bin/env bash
#SBATCH --job-name=llmtd-subgraph-stat
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/tabdiff/bin/python"
OUT="${ANALYSIS_RUN_ROOT:-$A/outputs}/subgroup_selector_stat_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/subgroup_selector_stat_audit.py" \
  --runs "$A/outputs/subgroup_llm_gpt41mini_1050324_0" "$A/outputs/subgroup_llm_gpt54mini_1050324_1" \
  --output "$OUT" --bootstrap 5000 --seed 20260929
