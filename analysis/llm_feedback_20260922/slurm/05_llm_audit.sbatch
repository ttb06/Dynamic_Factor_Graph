#!/usr/bin/env bash
#SBATCH --job-name=llmtd-llmaudit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/llm_feedback_audit.py" \
  --cases "$ANALYSIS/outputs/pilot_probe/feedback_cases.jsonl" \
  --env-file "$ROOT/EIPC/.env" \
  --output "$ANALYSIS/outputs/llm_audit" \
  --max-cases 9 \
  --repeats 2
