#!/usr/bin/env bash
# Screening sweep on three additional classification datasets.  All tasks use
# paired noise and the same common evaluator; one task per dataset probes a
# stronger or positive-factor variant beyond the common three-point grid.
#SBATCH --job-name=llmtd-xdata-more
#SBATCH --partition=prioritized
#SBATCH --array=0-11%12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
i=$SLURM_ARRAY_TASK_ID
GROUP=$((i / 4))
LOCAL=$((i % 4))

case "$GROUP" in
  0)
    DATASET=adult
    GRAPH="$A/outputs/extended_bscg_adult_1081004/graph.json"
    EVAL_GRAPH="$GRAPH"
    CONFIG="$A/outputs/extended_full_train_adult_1081005/config.pkl"
    CHECKPOINT="$A/outputs/extended_full_train_adult_1081005/ema_model_epoch_8000.pt"
    EDGE_IDS='bscg_001,bscg_003'
    ;;
  1)
    DATASET=magic
    GRAPH="$A/outputs/extended_bscg_magic_1081009/graph.json"
    EVAL_GRAPH="$GRAPH"
    CONFIG="$A/outputs/extended_full_train_magic_1081010/config.pkl"
    CHECKPOINT="$A/outputs/extended_full_train_magic_1081010/ema_model_epoch_8000.pt"
    EDGE_IDS='bscg_000,bscg_001,bscg_002,bscg_003,bscg_004'
    ;;
  2)
    DATASET=diabetes
    GRAPH="$A/outputs/bootstrap_conditional_graph_diabetes_1070443_2/graph.json"
    EVAL_GRAPH="$GRAPH"
    CONFIG="$A/outputs/full_train_diabetes_1048479/config.pkl"
    CHECKPOINT="$A/outputs/full_train_diabetes_1048479/ema_model_epoch_8000.pt"
    EDGE_IDS='bscg_011'
    ;;
esac

FACTOR_STRENGTH=0
TREES=0
case "$LOCAL" in
  0) LABEL=s050_mid; STRENGTH=.50; MIN_T=.10; MAX_T=.40; TRUST=.35;;
  1) LABEL=s100_mid; STRENGTH=1.00; MIN_T=.10; MAX_T=.40; TRUST=.50;;
  2) LABEL=s100_broad; STRENGTH=1.00; MIN_T=.05; MAX_T=.55; TRUST=.50;;
  3)
    if [[ "$DATASET" == magic ]]; then
      LABEL=positive_factor; STRENGTH=.50; MIN_T=.10; MAX_T=.40; TRUST=.35
      FACTOR_STRENGTH=.15; TREES=96
    else
      LABEL=s150_mid; STRENGTH=1.50; MIN_T=.10; MAX_T=.40; TRUST=.75
    fi
    ;;
esac

SEED=$((20261060 + i))
OUT="$A/outputs/extended_dynamic_${DATASET}_${LABEL}_${SLURM_ARRAY_JOB_ID}_${i}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset "$DATASET" --graph "$GRAPH" \
  --include-edge-ids "$EDGE_IDS" --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --samples 4096 --steps 50 --seed "$SEED" \
  --mode copula --strength "$FACTOR_STRENGTH" --passes 1 --trees "$TREES" \
  --min-r2 .02 --clip 1 --factor-min-t .15 --factor-max-t .55 \
  --factor-schedule middle_peak --interval-space log_sigma_quantile \
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

