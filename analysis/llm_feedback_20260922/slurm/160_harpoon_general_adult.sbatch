#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-general
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -uo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
RUN="$ROOT/baseline_runs/harpoon_general_adult_20261002"
PY="$ROOT/baseline_runs/envs/harpoon-official/bin/python"
cd "$RUN"

status=0
for constraint in range category both or; do
  echo "BEGIN method=Harpoon constraint=$constraint"
  if "$PY" sampling_harpoon_ohe_tubular_generalconstraints.py --dataname adult --constraint "$constraint" --num_trials 1 --gpu 0; then code=0; else code=$?; status=1; fi
  echo "END method=Harpoon constraint=$constraint exit=$code"
  echo "BEGIN method=RePaint constraint=$constraint"
  if "$PY" sampling_repaint_generalconstraints.py --dataname adult --constraint "$constraint" --num_trials 1 --gpu 0; then code=0; else code=$?; status=1; fi
  echo "END method=RePaint constraint=$constraint exit=$code"
done
echo "aggregate_status=$status (individual failures, if any, are retained in this log; the sweep continues by design)"
exit 0
