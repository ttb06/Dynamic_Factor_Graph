#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mixgraph-ft
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-4
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python

VARIANTS=(head_continue head_calibrated full_continue full_calibrated full_two_view)
SCOPES=(head head full full full)
LRS=(1e-5 1e-5 2e-6 2e-6 2e-6)
APP=(0 .10 0 .10 .10)
REL=(0 .10 0 .10 .10)
RATE=(0 2 0 2 2)
CONS=(0 0 0 0 .10)
i=$SLURM_ARRAY_TASK_ID
variant=${VARIANTS[$i]}

"$PY" "$A/scripts/finetune_mixed_graph_calibrated_tabdiff.py" \
  --repo "$REPO" --dataset default \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --epochs 300 --batch-size 4096 --learning-rate "${LRS[$i]}" \
  --trainable-scope "${SCOPES[$i]}" \
  --applicability-weight "${APP[$i]}" --relation-weight "${REL[$i]}" \
  --rate-weight "${RATE[$i]}" --consistency-weight "${CONS[$i]}" \
  --t-min .10 --t-max .50 --truth-temperature .20 --save-every 100 --seed 20261391 \
  --output "$A/outputs/mixed_graph_finetune_${variant}_${SLURM_ARRAY_JOB_ID}_${i}"
