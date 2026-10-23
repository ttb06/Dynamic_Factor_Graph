#!/usr/bin/env bash
# Focused follow-up after the first distribution-transport screen.  News tests
# whether the trust region and actuation time explain why online gains are
# smaller than the train-only post-hoc ceiling.  Default tests target-
# conditional transport with explicit tail preservation for Beta/coverage.
#SBATCH --job-name=llmtd-transport-geom
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

CONDITION=()
TAIL=0
if (( i < 8 )); then
  DATASET=news
  GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
  CONFIG="$A/outputs/full_train_news_official_regression_1093974/config.pkl"
  CHECKPOINT="$A/outputs/news_e2e_teacher_continue_full_1094488_0/train/ema_model_final.pt"
  SAMPLES=8192; SEED=20261100
  REGRESSION=(--regression-target-mode numerical)
  FAMILY=(--family-strengths llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5 --family-transports statistical_order_program=rank)
  EDGE_ARGS=()
  case "$i" in
    0) LABEL=s050_mid_tr150; STRENGTH=.50; TMIN=.20; TMAX=.35; TRUST=1.50; ONCE=1;;
    1) LABEL=s100_mid_tr150; STRENGTH=1.00; TMIN=.20; TMAX=.35; TRUST=1.50; ONCE=1;;
    2) LABEL=s100_mid_tr300; STRENGTH=1.00; TMIN=.20; TMAX=.35; TRUST=3.00; ONCE=1;;
    3) LABEL=s100_mid_unbounded; STRENGTH=1.00; TMIN=.20; TMAX=.35; TRUST=0; ONCE=1;;
    4) LABEL=s050_latemid_tr150; STRENGTH=.50; TMIN=.10; TMAX=.20; TRUST=1.50; ONCE=1;;
    5) LABEL=s100_latemid_tr150; STRENGTH=1.00; TMIN=.10; TMAX=.20; TRUST=1.50; ONCE=1;;
    6) LABEL=s005_repeat_tr150; STRENGTH=.05; TMIN=.10; TMAX=.40; TRUST=1.50; ONCE=0;;
    7) LABEL=s010_repeat_tr150; STRENGTH=.10; TMIN=.10; TMAX=.40; TRUST=1.50; ONCE=0;;
  esac
else
  DATASET=default
  GRAPH="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
  CONFIG="$A/outputs/full_train_default_1033760/config.pkl"
  CHECKPOINT="$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt"
  SAMPLES=4096; SEED=20261101
  REGRESSION=(); FAMILY=()
  EDGE_ARGS=(--include-edge-ids old::r1_001,old::r1_014,old::r1_015,old::r1_017)
  CONDITION=(--distribution-transport-condition-target)
  case "$i" in
     8) LABEL=cond_s025_once; STRENGTH=.25; TMIN=.20; TMAX=.35; TRUST=1.00; TAIL=0; ONCE=1;;
     9) LABEL=cond_s050_tail10; STRENGTH=.50; TMIN=.20; TMAX=.35; TRUST=1.00; TAIL=.10; ONCE=1;;
    10) LABEL=cond_s050_tail20; STRENGTH=.50; TMIN=.20; TMAX=.35; TRUST=1.50; TAIL=.20; ONCE=1;;
    11) LABEL=cond_s005_repeat_tail10; STRENGTH=.05; TMIN=.10; TMAX=.40; TRUST=1.00; TAIL=.10; ONCE=0;;
  esac
fi

ONCE_ARGS=(); if [[ "$ONCE" == 1 ]]; then ONCE_ARGS=(--distribution-transport-once); fi
OUT="$A/outputs/transport_geometry_${DATASET}_${LABEL}_${SLURM_ARRAY_JOB_ID}_${i}"

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
  --projection-persistence-steps 2 --projection-trust-radius "$TRUST" \
  --distribution-transport-mode copula --distribution-transport-scope all \
  --distribution-transport-strength "$STRENGTH" \
  --distribution-transport-min-t "$TMIN" --distribution-transport-max-t "$TMAX" \
  --distribution-transport-schedule middle_peak \
  --distribution-transport-preserve-tail "$TAIL" "${CONDITION[@]}" "${ONCE_ARGS[@]}" \
  --guidance-stage both --stochastic-sampler --second-order-correction \
  --output "$OUT/sample"

if [[ "$DATASET" == news ]]; then
  "$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
    --sample "Backbone=$OUT/sample/samples/unconditional.csv" \
    --sample "DistributionGraph=$OUT/sample/samples/dense_factor.csv" \
    --sample "DistributionGraph+Certifier=$OUT/sample/samples/dense_factor_then_certifier.csv" \
    --seed "$SEED" --skip-detection --output "$OUT/eval"
else
  "$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
    --repo "$REPO" --dataset "$DATASET" --graph "$GRAPH" \
    --sample "TabDiff=$OUT/sample/samples/unconditional.csv" \
    --sample "FinalRepair=$OUT/sample/samples/final_repair.csv" \
    --sample "DistributionGraph=$OUT/sample/samples/dense_factor.csv" \
    --sample "DistributionGraph+Certifier=$OUT/sample/samples/dense_factor_then_certifier.csv" \
    --seed "$SEED" --max-rows "$SAMPLES" --output "$OUT/eval"
fi
