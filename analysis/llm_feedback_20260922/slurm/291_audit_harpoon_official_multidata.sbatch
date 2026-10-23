#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-audit3
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
"$ROOT/baseline_runs/envs/grdm/bin/python" "$A/scripts/audit_harpoon_official_multidata.py" \
  --run-root "$ROOT/baseline_runs/harpoon_official_multidata_20261008" \
  --paper-csv "$ROOT/baseline_repos/harpoon/experiments/general_constraints_updated_utility.csv" \
  --output "$A/outputs/harpoon_official_multidata_${SLURM_JOB_ID}"
