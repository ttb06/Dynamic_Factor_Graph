#!/usr/bin/env bash
# Official ForestDiffusion on two additional matched splits.
#SBATCH --job-name=llmtd-more-forest
#SBATCH --partition=prioritized
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
DATASETS=(magic diabetes)
GRAPHS=("$A/outputs/extended_bscg_magic_1081009/graph.json" "$A/outputs/bootstrap_conditional_graph_diabetes_1070443_2/graph.json")
i=$SLURM_ARRAY_TASK_ID
DATASET=${DATASETS[$i]}
GRAPH=${GRAPHS[$i]}
SEED=$((20261080 + i))
OUT="$A/outputs/forestdiffusion_${DATASET}_official_k100_${SLURM_ARRAY_JOB_ID}_${i}"
cd "$REPO"
export PYTHONPATH="$A/optional_dependency_shims:$ROOT/baseline_repos/forestdiffusion/Python-Package/base-ForestDiffusion:${PYTHONPATH:-}"
"$PY" "$A/scripts/run_forestdiffusion_baseline.py" \
  --repo "$REPO" --forest-repo "$ROOT/baseline_repos/forestdiffusion" \
  --dataset "$DATASET" --duplicate-k 100 --n-t 50 --n-estimators 100 \
  --n-jobs 1 --n-batch 10 --gpu-hist --samples 4096 --seed "$SEED" --output "$OUT"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset "$DATASET" --graph "$GRAPH" \
  --sample "ForestDiffusion-official=$OUT/forest_diffusion.csv" \
  --seed "$SEED" --max-rows 4096 --output "$OUT/eval"

