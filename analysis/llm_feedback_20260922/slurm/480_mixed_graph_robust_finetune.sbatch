#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mixgraph-robust
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python

"$PY" "$A/scripts/finetune_mixed_graph_calibrated_tabdiff.py" \
  --repo "$REPO" --dataset default \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --epochs 150 --batch-size 4096 --learning-rate 2e-6 --trainable-scope full \
  --applicability-weight .10 --relation-weight .10 --rate-weight 2 --consistency-weight 0 \
  --calibration-target split_median \
  --t-min .10 --t-max .50 --truth-temperature .20 --save-every 150 --seed 20261391 \
  --output "$A/outputs/mixed_graph_robust_finetune_${SLURM_JOB_ID}"
