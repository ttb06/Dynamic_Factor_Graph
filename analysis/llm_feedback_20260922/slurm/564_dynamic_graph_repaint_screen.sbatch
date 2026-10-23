#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dg-repaint
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-11
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"

datasets=(shoppers shoppers shoppers shoppers diabetes diabetes diabetes diabetes default default default default)
graph_kinds=(bscg bscg bscg bscg bscg bscg bscg bscg bscg bscg bscg actionable)
triggers=(mid mid final mid mid mid final mid mid mid final mid)
times=(.25 .40 .25 .25 .25 .40 .25 .25 .25 .40 .25 .25)
rates=(.01 .01 .01 .05 .01 .01 .01 .05 .05 .05 .05 .03)
labels=(mid25_hard mid40_hard final_hard mid25_broad mid25_hard mid40_hard final_hard mid25_broad mid25_broad mid40_broad final_broad actionable_mid25)

i=$SLURM_ARRAY_TASK_ID; dataset=${datasets[$i]}; graph_kind=${graph_kinds[$i]}
trigger=${triggers[$i]}; checkpoint_time=${times[$i]}; rate=${rates[$i]}; label=${labels[$i]}
case "$dataset" in
  shoppers) source_run=full_train_shoppers_1033761; graph_index=1 ;;
  diabetes) source_run=full_train_diabetes_1048479; graph_index=2 ;;
  default) source_run=full_train_default_1033760; graph_index=0 ;;
esac
if [[ "$graph_kind" == bscg ]]; then
  graph="$A/outputs/bootstrap_conditional_graph_${dataset}_1070443_${graph_index}/graph.json"
else
  case "$dataset" in shoppers) actionable_index=0 ;; default) actionable_index=1 ;; diabetes) actionable_index=2 ;; esac
  graph="$A/outputs/actionable_graph_v2_${dataset}_1060952_${actionable_index}/empirical_graph.json"
fi
cd "$REPO"
"$PY" "$A/scripts/dynamic_graph_conditional_repaint.py" \
  --repo "$REPO" --dataset "$dataset" --graph "$graph" \
  --config "$A/outputs/$source_run/config.pkl" \
  --checkpoint "$A/outputs/$source_run/ema_model_epoch_8000.pt" \
  --trigger "$trigger" --checkpoint-time "$checkpoint_time" \
  --max-calibration-rate "$rate" --branches 4 --rounds 2 --samples 2048 --steps 50 \
  --seed 20261440 \
  --output "$A/outputs/dg_repaint_${dataset}_${label}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
