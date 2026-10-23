#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mixgraph-smoke
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-3
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python

VARIANTS=(continue relation calibrated two_view)
APP=(0 .10 .10 .10)
REL=(0 .10 .10 .10)
RATE=(0 0 2 2)
CONS=(0 0 0 .10)
i=$SLURM_ARRAY_TASK_ID
variant=${VARIANTS[$i]}

"$PY" "$A/scripts/finetune_mixed_graph_calibrated_tabdiff.py" \
  --repo "$REPO" --dataset default \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --epochs 5 --batch-size 4096 --learning-rate 1e-5 --trainable-scope head \
  --applicability-weight "${APP[$i]}" --relation-weight "${REL[$i]}" \
  --rate-weight "${RATE[$i]}" --consistency-weight "${CONS[$i]}" \
  --t-min .10 --t-max .50 --truth-temperature .20 --save-every 5 --seed 20261390 \
  --output "$A/outputs/mixed_graph_smoke_${variant}_${SLURM_ARRAY_JOB_ID}_${i}"
