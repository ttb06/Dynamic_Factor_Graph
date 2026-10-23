#!/usr/bin/env bash
# Focused follow-up: a small number of separated distribution pulses rather
# than repeated guidance at every step, plus scale-separated categorical
# factor messages on Default.
#SBATCH --job-name=llmtd-pulsed-factors
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
REGRESSION=(); FAMILY=(); EDGE_ARGS=()
CAT=(--categorical-guidance-mode none)

if (( i < 8 )); then
  DATASET=news
  GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
  CONFIG="$A/outputs/full_train_news_official_regression_1093974/config.pkl"
  CHECKPOINT="$A/outputs/news_e2e_teacher_continue_full_1094488_0/train/ema_model_final.pt"
  SAMPLES=8192; SEED=20261130
  REGRESSION=(--regression-target-mode numerical)
  FAMILY=(--family-strengths llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5 --family-transports statistical_order_program=rank)
  MODE=copula; QBINS=4; CALLS=2; GAP=.18
  case "$i" in
    0) LABEL=q04_s005_2pulse; TSTRENGTH=.05;;
    1) LABEL=q04_s010_2pulse; TSTRENGTH=.10;;
    2) LABEL=q04_s015_2pulse; TSTRENGTH=.15;;
    3) LABEL=q04_s025_2pulse; TSTRENGTH=.25;;
    4) LABEL=q04_s010_2pulse_gap12; TSTRENGTH=.10; GAP=.12;;
    5) LABEL=q04_s010_3pulse; TSTRENGTH=.10; CALLS=3; GAP=.12;;
    6) LABEL=global_s025_2pulse; TSTRENGTH=.25; QBINS=0;;
    7) LABEL=global_s050_2pulse; TSTRENGTH=.50; QBINS=0;;
  esac
  TRANSPORT=(--distribution-transport-mode "$MODE" --distribution-transport-scope all --distribution-transport-strength "$TSTRENGTH" --distribution-transport-max-calls "$CALLS" --distribution-transport-min-call-gap "$GAP")
  if (( QBINS > 1 )); then
    TRANSPORT+=(--distribution-transport-condition-target-quantiles "$QBINS")
  fi
else
  DATASET=default
  GRAPH="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
  CONFIG="$A/outputs/full_train_default_1033760/config.pkl"
  CHECKPOINT="$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt"
  SAMPLES=4096; SEED=20261131
  EDGE_ARGS=(--include-edge-ids old::r1_001,old::r1_014,old::r1_015,old::r1_017)
  TRANSPORT=(--distribution-transport-mode none)
  case "$i" in
     8) LABEL=cat_target_s025_pair2; CAT=(--categorical-guidance-mode target --categorical-guidance-strength .25 --categorical-guidance-pair-scale 2);;
     9) LABEL=cat_target_s050_pair2; CAT=(--categorical-guidance-mode target --categorical-guidance-strength .50 --categorical-guidance-pair-scale 2);;
    10) LABEL=cat_marginal_s010; CAT=(--categorical-guidance-mode marginal --categorical-guidance-strength .10);;
    11) LABEL=cat_hybrid_scaled_condnum; CAT=(--categorical-guidance-mode marginal_target --categorical-guidance-strength .25 --categorical-guidance-marginal-scale .10 --categorical-guidance-pair-scale 2); TRANSPORT=(--distribution-transport-mode copula --distribution-transport-scope all --distribution-transport-strength .50 --distribution-transport-condition-target --distribution-transport-preserve-tail .10 --distribution-transport-once);;
  esac
fi

OUT="$A/outputs/pulsed_factors_${DATASET}_${LABEL}_${SLURM_ARRAY_JOB_ID}_${i}"
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
  --distribution-transport-min-t .10 --distribution-transport-max-t .40 \
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
