#!/usr/bin/env bash
# Train-only distribution-gap transport inside the reverse trajectory.  The
# batch operator aligns numeric marginals/copula at a bounded middle state;
# typed dynamic graph projection is applied in the same sampler.  All methods
# in each task use paired noise and the evaluator never enters the controller.
#SBATCH --job-name=llmtd-distgraph-mid
#SBATCH --partition=prioritized
#SBATCH --array=0-11%12
#SBATCH --gres=gpu:1
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

# Four focused operating points per dataset.  The first three test a
# full-numeric statistical graph; the fourth restricts transport to verified
# semantic graph components.
LOCAL=$((i % 4))
GROUP=$((i / 4))
case "$LOCAL" in
  0) LABEL=copula_s010_once; TMODE=copula; TSCOPE=all; TSTRENGTH=.10; TMIN=.20; TMAX=.35; TONCE=1;;
  1) LABEL=copula_s025_once; TMODE=copula; TSCOPE=all; TSTRENGTH=.25; TMIN=.20; TMAX=.35; TONCE=1;;
  2) LABEL=copula_s050_once; TMODE=copula; TSCOPE=all; TSTRENGTH=.50; TMIN=.20; TMAX=.35; TONCE=1;;
  3) LABEL=graphcopula_s050_once; TMODE=graph_copula; TSCOPE=graph; TSTRENGTH=.50; TMIN=.20; TMAX=.35; TONCE=1;;
esac

case "$GROUP" in
  0)
    DATASET=news
    GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
    CONFIG="$A/outputs/full_train_news_official_regression_1093974/config.pkl"
    CHECKPOINT="$A/outputs/news_e2e_teacher_continue_full_1094488_0/train/ema_model_final.pt"
    SAMPLES=8192
    SEED=20261080
    REGRESSION=(--regression-target-mode numerical)
    FAMILY=(--family-strengths llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5 --family-transports statistical_order_program=rank)
    EDGE_ARGS=()
    ;;
  1)
    DATASET=default
    GRAPH="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
    CONFIG="$A/outputs/full_train_default_1033760/config.pkl"
    CHECKPOINT="$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt"
    SAMPLES=4096
    SEED=20261084
    REGRESSION=()
    FAMILY=()
    EDGE_ARGS=(--include-edge-ids old::r1_001,old::r1_014,old::r1_015,old::r1_017)
    ;;
  2)
    DATASET=magic
    GRAPH="$A/outputs/extended_bscg_magic_1081009/graph.json"
    CONFIG="$A/outputs/extended_full_train_magic_1081010/config.pkl"
    CHECKPOINT="$A/outputs/extended_full_train_magic_1081010/ema_model_epoch_8000.pt"
    SAMPLES=4096
    SEED=20261088
    REGRESSION=()
    FAMILY=()
    EDGE_ARGS=(--include-edge-ids bscg_000,bscg_001,bscg_002,bscg_003,bscg_004)
    ;;
esac

OUT="$A/outputs/distribution_graph_middle_${DATASET}_${LABEL}_${SLURM_ARRAY_JOB_ID}_${i}"
ONCE_ARGS=()
if [[ "$TONCE" == 1 ]]; then ONCE_ARGS=(--distribution-transport-once); fi

cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset "$DATASET" "${REGRESSION[@]}" \
  --graph "$GRAPH" "${EDGE_ARGS[@]}" --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --samples "$SAMPLES" --steps 50 --seed "$SEED" \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat "${FAMILY[@]}" \
  --projection-strength 1 --projection-min-t .10 --projection-max-t .40 \
  --projection-interval-space edge_log_sigma_quantile --projection-schedule middle_peak \
  --projection-route-mode auto_edge_valley_isotonic \
  --projection-stability-mode rank --projection-stability-threshold .001 \
  --projection-persistence-steps 2 --projection-trust-radius .75 \
  --distribution-transport-mode "$TMODE" --distribution-transport-scope "$TSCOPE" \
  --distribution-transport-strength "$TSTRENGTH" \
  --distribution-transport-min-t "$TMIN" --distribution-transport-max-t "$TMAX" \
  --distribution-transport-schedule flat "${ONCE_ARGS[@]}" \
  --guidance-stage both --stochastic-sampler --second-order-correction \
  --output "$OUT/sample"

if [[ "$DATASET" == news ]]; then
  "$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
    --sample "TabDiff=$OUT/sample/samples/unconditional.csv" \
    --sample "DynamicGraph=$OUT/sample/samples/dense_factor.csv" \
    --sample "DynamicGraph+Certifier=$OUT/sample/samples/dense_factor_then_certifier.csv" \
    --seed "$SEED" --skip-detection --output "$OUT/eval"
else
  "$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
    --repo "$REPO" --dataset "$DATASET" --graph "$GRAPH" \
    --sample "TabDiff=$OUT/sample/samples/unconditional.csv" \
    --sample "FinalRepair=$OUT/sample/samples/final_repair.csv" \
    --sample "DynamicGraph=$OUT/sample/samples/dense_factor.csv" \
    --sample "DynamicGraph+Certifier=$OUT/sample/samples/dense_factor_then_certifier.csv" \
    --seed "$SEED" --max-rows "$SAMPLES" --output "$OUT/eval"
fi
