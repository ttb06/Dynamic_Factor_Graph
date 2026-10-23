#!/usr/bin/env bash
# CTGAN/TVAE anchors on Magic and Diabetes with the common local protocol.
#SBATCH --job-name=llmtd-more-sdv
#SBATCH --partition=prioritized
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
DATASETS=(magic magic diabetes diabetes)
METHODS=(ctgan tvae ctgan tvae)
GRAPHS=("$A/outputs/extended_bscg_magic_1081009/graph.json" "$A/outputs/extended_bscg_magic_1081009/graph.json" "$A/outputs/bootstrap_conditional_graph_diabetes_1070443_2/graph.json" "$A/outputs/bootstrap_conditional_graph_diabetes_1070443_2/graph.json")
i=$SLURM_ARRAY_TASK_ID
DATASET=${DATASETS[$i]}
METHOD=${METHODS[$i]}
GRAPH=${GRAPHS[$i]}
SEED=$((20261084 + i))
OUT="$A/outputs/sdv_${DATASET}_${METHOD}_${SLURM_ARRAY_JOB_ID}_${i}"
cd "$REPO"
"$PY" "$A/scripts/train_sdv_shoppers_baseline.py" \
  --repo "$REPO" --dataset "$DATASET" --method "$METHOD" --epochs 300 \
  --samples 4096 --seed "$SEED" --output "$OUT"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset "$DATASET" --graph "$GRAPH" \
  --sample "${METHOD^^}-SDV=$OUT/synthetic.csv" \
  --seed "$SEED" --max-rows 4096 --output "$OUT/eval"

