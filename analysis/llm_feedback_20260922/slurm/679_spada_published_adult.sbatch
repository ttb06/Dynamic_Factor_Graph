#!/usr/bin/env bash
#SBATCH --job-name=llmtd-spada25-adult
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/sage-official/bin/python"
export LD_LIBRARY_PATH="$ROOT/baseline_runs/envs/sage-official/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$REPO"
"$PY" "$A/scripts/run_spada_published_baseline.py" \
  --spada-repo "$ROOT/baseline_repos/spada" --train "$REPO/data/adult/train.csv" \
  --output "$A/outputs/spada_published_adult_${SLURM_JOB_ID}" \
  --samples 4096 --epochs 5000 --patience 500 --seed 20261016
