#!/usr/bin/env bash
# ForestDiffusion (AISTATS 2024) using the authors' Python package and the
# full duplicate-k=100 setting used in our completed Default/Shoppers runs.
#SBATCH --job-name=llmtd-forest24-news
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
OUT="$A/outputs/forestdiffusion_news_official_k100_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$A/optional_dependency_shims:$ROOT/baseline_repos/forestdiffusion/Python-Package/base-ForestDiffusion:${PYTHONPATH:-}"
"$PY" "$A/scripts/run_forestdiffusion_baseline.py" \
  --repo "$REPO" --forest-repo "$ROOT/baseline_repos/forestdiffusion" \
  --dataset news --duplicate-k 100 --n-t 50 --n-estimators 100 \
  --n-jobs 1 --n-batch 10 --gpu-hist --samples 4096 --seed 20261017 \
  --output "$OUT"
