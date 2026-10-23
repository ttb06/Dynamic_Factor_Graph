#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cuts-german
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=30:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CUTS_MODE:?Set CUTS_MODE to baseline or constrained}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
CUTS="$ROOT/external/cuts"
PY="$ROOT/baseline_runs/envs/cuts/bin/python"
cd "$CUTS"
COMMON=(--dataset German --n_samples 1 --n_resamples 1 --random_seed 42 --workload all_three_with_labels --epsilon 0)
if [[ "$CUTS_MODE" == baseline ]]; then
  "$PY" run_constraint_program_benchmark.py "${COMMON[@]}" --baseline_mode
elif [[ "$CUTS_MODE" == constrained ]]; then
  "$PY" run_constraint_program_benchmark.py "${COMMON[@]}"
else
  echo "Unknown CUTS_MODE=$CUTS_MODE" >&2
  exit 2
fi
