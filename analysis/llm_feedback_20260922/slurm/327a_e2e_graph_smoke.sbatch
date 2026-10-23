#!/usr/bin/env bash
#SBATCH --job-name=llmtd-e2e-smoke
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"; PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
"$PY" "$A/scripts/finetune_graph_consistent_tabdiff.py" --repo "$REPO" --dataset shoppers \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --graph "$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json" \
  --epochs 1 --batch-size 4096 --learning-rate 1e-5 --graph-weight .05 --t-min .15 --t-max .60 \
  --truth-temperature .20 --save-every 1 --seed 20261230 \
  --output "$A/outputs/e2e_graph_smoke_${SLURM_JOB_ID}"
