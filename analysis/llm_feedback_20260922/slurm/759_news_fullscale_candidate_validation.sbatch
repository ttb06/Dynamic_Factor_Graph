#!/usr/bin/env bash
# Full News train-size validation for the two Pareto candidates and the two
# most informative source ablations. Includes the slower detection metric.
#SBATCH --job-name=llmtd-news-fullscale
#SBATCH --partition=prioritized
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
CONFIG="$A/outputs/full_train_news_official_regression_1093974/config.pkl"
CKPT="$A/outputs/full_train_news_official_regression_1093974/ema_model_epoch_8000.pt"
i=$SLURM_ARRAY_TASK_ID

LABEL=""; TRANSPORT=direct; FAMILY_TRANSPORTS=statistical_order_program=rank
WEIGHTS=llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5
case "$i" in
  0) LABEL=main_hybrid;;
  1) LABEL=rank_all; TRANSPORT=rank; FAMILY_TRANSPORTS="";;
  2) LABEL=main_without_llm;
     WEIGHTS=llm_directional_bound=0,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5;;
  3) LABEL=main_without_order;
     WEIGHTS=llm_directional_bound=1,statistical_order_program=0,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5;;
esac

SEED=20261023
OUT="$A/outputs/news_fullscale_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
ARGS=(
  --repo "$REPO" --dataset news --regression-target-mode numerical
  --graph "$GRAPH" --config "$CONFIG" --checkpoint "$CKPT"
  --samples 35679 --steps 50 --seed "$SEED"
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat
  --program-transport "$TRANSPORT" --family-strengths "$WEIGHTS"
  --projection-strength 1 --projection-min-t .10 --projection-max-t .40
  --projection-interval-space edge_log_sigma_quantile --projection-schedule middle_peak
  --projection-route-mode auto_edge_valley_isotonic --projection-stability-mode rank
  --projection-stability-threshold .001 --projection-persistence-steps 2
  --projection-trust-radius .50 --guidance-stage both
  --stochastic-sampler --second-order-correction --output "$OUT"
)
if [[ -n "$FAMILY_TRANSPORTS" ]]; then ARGS+=(--family-transports "$FAMILY_TRANSPORTS"); fi

cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" "${ARGS[@]}"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "matched_final_repair=$OUT/samples/final_repair.csv" \
  --sample "dynamic=$OUT/samples/dense_factor.csv" \
  --sample "dynamic_then_certifier=$OUT/samples/dense_factor_then_certifier.csv" \
  --seed "$SEED" --output "$OUT/eval_full"
