#!/usr/bin/env bash
#SBATCH --job-name=llmtd-lcats26-full
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
LCATS="$ROOT/baseline_repos/lcats"
PY="$ROOT/baseline_runs/envs/lcats-official/bin/python"
export LCATS_ROOT="$LCATS"
export TABSYN_ROOT="$LCATS"
export PYTHONPATH="$LCATS:$REPO:$A/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$LCATS"

# Reuse the complete, API-verified phase-1 package on retries.  A failure in
# TabSyn must not repeat paid API calls.
if [[ ! -s "$LCATS/data/insurance/constraints.json" || \
      ! -s "$LCATS/data/insurance/metadata.json" || \
      ! -s "$LCATS/data/insurance/cbg_list.json" || \
      ! -s "$LCATS/data/insurance/verify_script.py" ]]; then
  "$PY" "$A/scripts/run_with_unorouter.py" \
    --env-file "$ROOT/.env" --model gpt-5.6-luna \
    "$PY" "$LCATS/LCATS/phase_1.py" --dataname insurance
else
  echo "Reusing verified LCATS phase-1 artifacts; no API call needed."
fi
"$PY" "$LCATS/LCATS/phase_2.py" --dataname insurance
"$PY" "$LCATS/LCATS/phase_3.py" --dataname insurance
"$PY" "$A/scripts/run_with_unorouter.py" \
  --env-file "$ROOT/.env" --model gpt-5.6-luna \
  "$PY" "$LCATS/LCATS/phase_4.py" --dataname insurance
"$PY" "$LCATS/LCATS/phase_5.py" --dataname insurance
