#!/usr/bin/env bash
# Mixed-type distribution messages inside a bounded middle interval.
# Default tests categorical log-density messages, alone and together with
# target-conditional numeric copula transport.  News tests row/state-dynamic
# regression-target quantile factors.  Every task uses paired backbone noise.
#SBATCH --job-name=llmtd-mixed-distmsg
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

TRANSPORT=(--distribution-transport-mode none)
CAT=(--categorical-guidance-mode none)
REGRESSION=()
FAMILY=()
EDGE_ARGS=()

if (( i < 6 )); then
  DATASET=default
  GRAPH="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
  CONFIG="$A/outputs/full_train_default_1033760/config.pkl"
  CHECKPOINT="$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt"
  SAMPLES=4096; SEED=20261120
  EDGE_ARGS=(--include-edge-ids old::r1_001,old::r1_014,old::r1_015,old::r1_017)
  case "$i" in
    0) LABEL=cat_marginal_s025; CAT=(--categorical-guidance-mode marginal --categorical-guidance-strength .25);;
    1) LABEL=cat_target_s025; CAT=(--categorical-guidance-mode target --categorical-guidance-strength .25);;
    2) LABEL=cat_margtarget_s010; CAT=(--categorical-guidance-mode marginal_target --categorical-guidance-strength .10);;
    3) LABEL=cat_margtarget_s025_condnum; CAT=(--categorical-guidance-mode marginal_target --categorical-guidance-strength .25); TRANSPORT=(--distribution-transport-mode copula --distribution-transport-scope all --distribution-transport-strength .50 --distribution-transport-condition-target --distribution-transport-preserve-tail .10 --distribution-transport-once);;
    4) LABEL=cat_margtarget_s050_condnum; CAT=(--categorical-guidance-mode marginal_target --categorical-guidance-strength .50); TRANSPORT=(--distribution-transport-mode copula --distribution-transport-scope all --distribution-transport-strength .50 --distribution-transport-condition-target --distribution-transport-preserve-tail .10 --distribution-transport-once);;
    5) LABEL=cat_margpair_s025_condnum; CAT=(--categorical-guidance-mode marginal_pairwise --categorical-guidance-strength .25); TRANSPORT=(--distribution-transport-mode copula --distribution-transport-scope all --distribution-transport-strength .50 --distribution-transport-condition-target --distribution-transport-preserve-tail .10 --distribution-transport-once);;
  esac
else
  DATASET=news
  GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
  CONFIG="$A/outputs/full_train_news_official_regression_1093974/config.pkl"
  CHECKPOINT="$A/outputs/news_e2e_teacher_continue_full_1094488_0/train/ema_model_final.pt"
  SAMPLES=8192; SEED=20261121
  REGRESSION=(--regression-target-mode numerical)
  FAMILY=(--family-strengths llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5 --family-transports statistical_order_program=rank)
  case "$i" in
     6) LABEL=q04_s010; QBINS=4; TSTRENGTH=.10;;
     7) LABEL=q04_s025; QBINS=4; TSTRENGTH=.25;;
     8) LABEL=q04_s050; QBINS=4; TSTRENGTH=.50;;
     9) LABEL=q08_s010; QBINS=8; TSTRENGTH=.10;;
    10) LABEL=q08_s025; QBINS=8; TSTRENGTH=.25;;
    11) LABEL=q08_s050; QBINS=8; TSTRENGTH=.50;;
  esac
  TRANSPORT=(--distribution-transport-mode copula --distribution-transport-scope all --distribution-transport-strength "$TSTRENGTH" --distribution-transport-condition-target-quantiles "$QBINS" --distribution-transport-once)
fi

OUT="$A/outputs/mixed_distribution_messages_${DATASET}_${LABEL}_${SLURM_ARRAY_JOB_ID}_${i}"
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
  --distribution-transport-min-t .20 --distribution-transport-max-t .35 \
  --distribution-transport-schedule flat "${TRANSPORT[@]}" \
  --categorical-guidance-min-t .20 --categorical-guidance-max-t .50 \
  --categorical-guidance-schedule middle_peak "${CAT[@]}" \
  --guidance-stage both --stochastic-sampler --second-order-correction \
  --output "$OUT/sample"

if [[ "$DATASET" == news ]]; then
  "$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
    --sample "Backbone=$OUT/sample/samples/unconditional.csv" \
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
