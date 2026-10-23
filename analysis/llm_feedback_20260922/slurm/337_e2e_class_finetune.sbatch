#!/usr/bin/env bash
#SBATCH --job-name=llmtd-e2e-class-ft
#SBATCH --partition=prioritized
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
VARIANTS=(hard5 hard10 near5); WEIGHTS=(5 10 5); CLASSES=(hard hard near_hard)
i=$SLURM_ARRAY_TASK_ID; variant=${VARIANTS[$i]}
"$PY" "$A/scripts/finetune_graph_consistent_tabdiff.py" --repo "$REPO" --dataset shoppers \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --graph "$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json" \
  --epochs 500 --batch-size 4096 --learning-rate 1e-5 --graph-weight "${WEIGHTS[$i]}" \
  --t-min .15 --t-max .60 --truth-temperature .20 --allowed-classifications "${CLASSES[$i]}" \
  --seed 20261303 --output "$A/outputs/e2e_class_finetune_${variant}_${SLURM_ARRAY_JOB_ID}_${i}"
