#!/usr/bin/env bash
# Conventional CTGAN/TVAE anchors on exactly the same Adult split/evaluator.
#SBATCH --job-name=llmtd-adult-sdv
#SBATCH --partition=prioritized
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
METHODS=(ctgan tvae)
i=$SLURM_ARRAY_TASK_ID
METHOD=${METHODS[$i]}
SEED=$((20261050 + i))
OUT="$A/outputs/sdv_adult_${METHOD}_${SLURM_ARRAY_JOB_ID}_${i}"
cd "$REPO"
"$PY" "$A/scripts/train_sdv_shoppers_baseline.py" \
  --repo "$REPO" --dataset adult --method "$METHOD" --epochs 300 \
  --samples 4096 --seed "$SEED" --output "$OUT"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset adult \
  --graph "$A/outputs/extended_bscg_adult_1081004/graph.json" \
  --sample "${METHOD^^}-SDV=$OUT/synthetic.csv" \
  --seed "$SEED" --max-rows 4096 --output "$OUT/eval"

