#!/usr/bin/env bash
# Official ForestDiffusion baseline on the repository Adult split.
#SBATCH --job-name=llmtd-adult-forest
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
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
OUT="$A/outputs/forestdiffusion_adult_official_k100_${SLURM_JOB_ID}"
cd "$REPO"
export PYTHONPATH="$A/optional_dependency_shims:$ROOT/baseline_repos/forestdiffusion/Python-Package/base-ForestDiffusion:${PYTHONPATH:-}"
"$PY" "$A/scripts/run_forestdiffusion_baseline.py" \
  --repo "$REPO" --forest-repo "$ROOT/baseline_repos/forestdiffusion" \
  --dataset adult --duplicate-k 100 --n-t 50 --n-estimators 100 \
  --n-jobs 1 --n-batch 10 --gpu-hist --samples 4096 --seed 20261048 --output "$OUT"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset adult \
  --graph "$A/outputs/extended_bscg_adult_1081004/graph.json" \
  --sample "ForestDiffusion-official=$OUT/forest_diffusion.csv" \
  --seed 20261048 --max-rows 4096 --output "$OUT/eval"

