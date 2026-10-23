#!/usr/bin/env bash
#SBATCH --job-name=llmtd-lcats26-eval
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${LCATS_FULL_JOB_ID:?LCATS_FULL_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
LCATS="$ROOT/baseline_repos/lcats"
PY="$ROOT/baseline_runs/envs/lcats-eval-api/bin/python"
export LCATS_ROOT="$LCATS"
export TABSYN_ROOT="$LCATS"
export PYTHONPATH="$LCATS:$REPO:$A/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$LCATS"
"$PY" "$A/scripts/run_with_unorouter.py" \
  --env-file "$ROOT/.env" --model gpt-5.6-luna \
  "$PY" "$LCATS/LCATS/phase_6.py" --dataname insurance

OUT="$A/outputs/lcats_published_full_audit_${SLURM_JOB_ID}"
"$PY" "$A/scripts/audit_lcats_official_outputs.py" \
  --lcats-root "$LCATS" --dataset insurance --output "$OUT"
