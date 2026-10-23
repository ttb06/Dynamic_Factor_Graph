#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
cd "$REPO"
"$ROOT/baseline_runs/envs/grdm/bin/python" "$A/scripts/audit_harpoon_official_general.py" \
  --run-root "$ROOT/baseline_runs/harpoon_general_adult_20261002" \
  --log "$A/logs/llmtd-harpoon-general_1052793.out" \
  --output "$A/outputs/harpoon_official_general_audit_${SLURM_JOB_ID}"
