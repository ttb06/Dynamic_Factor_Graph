#!/usr/bin/env bash
#SBATCH --job-name=llmtd-profile
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/tabdiff/bin/python"
cd "$REPO"
"$PY" - <<'PY'
from download_dataset import download_from_uci
for name in ("default", "shoppers"):
    download_from_uci(name)
PY
"$PY" process_dataset.py --dataname default
"$PY" process_dataset.py --dataname shoppers
"$PY" "$ANALYSIS/scripts/profile_and_rules.py" \
  --repo "$REPO" \
  --output "$RUN_ROOT/profile" \
  --datasets adult default shoppers
