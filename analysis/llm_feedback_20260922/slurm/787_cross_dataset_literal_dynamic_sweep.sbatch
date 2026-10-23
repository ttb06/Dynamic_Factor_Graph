#!/usr/bin/env bash
# Cross-dataset transfer of the best bounded middle-trajectory controller.
# The array tests a small predeclared strength/window grid on Default and
# Shoppers.  Each task emits paired TabDiff, final-repair, and dynamic samples.
#SBATCH --job-name=llmtd-lit-xdata
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-7%8
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
i=$SLURM_ARRAY_TASK_ID

if (( i < 4 )); then
  DATASET=default
  LOCAL=$i
  GRAPH="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
  EVAL_GRAPH="$GRAPH"
  CONFIG="$A/outputs/full_train_default_1033760/config.pkl"
  CHECKPOINT="$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt"
  EDGE_IDS='old::r1_001,old::r1_014,old::r1_015,old::r1_017'
else
  DATASET=shoppers
  LOCAL=$((i - 4))
  GRAPH="$A/outputs/shoppers_dynamic_confirm_s20261381_frozen_state_1067703_3/final_graph.json"
  EVAL_GRAPH="$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json"
  CONFIG="$A/outputs/full_train_shoppers_1033761/config.pkl"
  CHECKPOINT="$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt"
  EDGE_IDS='r1_014,seed_003,r2_006,r1_015,r1_005,r1_021,r1_013'
fi

case "$LOCAL" in
  0) LABEL=s050_mid; STRENGTH=.50; MIN_T=.10; MAX_T=.40; TRUST=.35;;
  1) LABEL=s100_mid; STRENGTH=1.00; MIN_T=.10; MAX_T=.40; TRUST=.50;;
  2) LABEL=s150_mid; STRENGTH=1.50; MIN_T=.10; MAX_T=.40; TRUST=.75;;
  3) LABEL=s100_broad; STRENGTH=1.00; MIN_T=.05; MAX_T=.55; TRUST=.50;;
esac

SEED=$((20261040 + i))
OUT="$A/outputs/literal_dynamic_${DATASET}_${LABEL}_${SLURM_ARRAY_JOB_ID}_${i}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset "$DATASET" --graph "$GRAPH" \
  --include-edge-ids "$EDGE_IDS" --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --samples 4096 --steps 50 --seed "$SEED" \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --projection-strength "$STRENGTH" --projection-min-t "$MIN_T" \
  --projection-max-t "$MAX_T" --projection-interval-space edge_log_sigma_quantile \
  --projection-schedule middle_peak --projection-route-mode auto_edge_valley_isotonic \
  --projection-stability-mode rank --projection-stability-threshold .001 \
  --projection-persistence-steps 2 --projection-trust-radius "$TRUST" \
  --guidance-stage both --stochastic-sampler --second-order-correction \
  --output "$OUT/sample"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset "$DATASET" --graph "$EVAL_GRAPH" \
  --sample "TabDiff=$OUT/sample/samples/unconditional.csv" \
  --sample "FinalRepair=$OUT/sample/samples/final_repair.csv" \
  --sample "DynamicGraph=$OUT/sample/samples/dense_factor.csv" \
  --sample "DynamicGraph+Certifier=$OUT/sample/samples/dense_factor_then_certifier.csv" \
  --seed "$SEED" --max-rows 4096 --output "$OUT/eval"

