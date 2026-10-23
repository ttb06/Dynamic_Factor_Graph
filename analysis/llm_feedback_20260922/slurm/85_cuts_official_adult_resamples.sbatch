#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cuts-r5
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CUTS_MODE:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
CUTS="$ROOT/external/cuts"
PY="$ROOT/baseline_runs/envs/cuts/bin/python"
cd "$CUTS"
COMMON=(--dataset ADULT --n_samples 1 --n_resamples 5 --random_seed 42 --workload all_three_with_labels --epsilon 0)
if [ "$CUTS_MODE" = baseline ]; then
  "$PY" run_constraint_program_benchmark.py "${COMMON[@]}" --baseline_mode
elif [ "$CUTS_MODE" = constrained ]; then
  "$PY" run_constraint_program_benchmark.py "${COMMON[@]}"
else
  echo "unknown CUTS_MODE=$CUTS_MODE" >&2
  exit 2
fi
