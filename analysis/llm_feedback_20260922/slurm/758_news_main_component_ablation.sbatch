#!/usr/bin/env bash
# Exact component ablation around the first common-metric Pareto operating
# point. One shared seed, 8192 samples, no online LLM/API.
#SBATCH --job-name=llmtd-news-main-ablation
#SBATCH --partition=prioritized
#SBATCH --array=0-11%12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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

LABEL=""; ROUTE=auto_edge_valley_isotonic; TRANSPORT=direct
FAMILY_TRANSPORTS=statistical_order_program=rank
PERSIST=2; TRUST=.50; PMIN=.10; PMAX=.40; SCHEDULE=middle_peak
WEIGHTS=llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5

case "$i" in
  0) LABEL=main_hybrid_valley_middle_trust;;
  1) LABEL=without_llm_edges;
     WEIGHTS=llm_directional_bound=0,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5;;
  2) LABEL=llm_edges_only;
     WEIGHTS=llm_directional_bound=1,statistical_order_program=0,statistical_functional_program=0,statistical_sum_program=0,statistical_conditional_sum_program=0;;
  3) LABEL=without_order_edges;
     WEIGHTS=llm_directional_bound=1,statistical_order_program=0,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5;;
  4) LABEL=order_edges_only;
     WEIGHTS=llm_directional_bound=0,statistical_order_program=.15,statistical_functional_program=0,statistical_sum_program=0,statistical_conditional_sum_program=0;;
  5) LABEL=without_algebraic_edges;
     WEIGHTS=llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=0,statistical_sum_program=0,statistical_conditional_sum_program=0;;
  6) LABEL=without_valley_anchor; ROUTE=auto_edge_isotonic;;
  7) LABEL=without_trust_region; TRUST=0;;
  8) LABEL=without_persistence; PERSIST=1;;
  9) LABEL=direct_order_action; FAMILY_TRANSPORTS="";;
  10) LABEL=rank_all_programs; TRANSPORT=rank; FAMILY_TRANSPORTS="";;
  11) LABEL=late_interval; PMIN=0; PMAX=.15; SCHEDULE=flat;;
esac

SEED=20261022
OUT="$A/outputs/news_main_ablation_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
ARGS=(
  --repo "$REPO" --dataset news --regression-target-mode numerical
  --graph "$GRAPH" --config "$CONFIG" --checkpoint "$CKPT"
  --samples 8192 --steps 50 --seed "$SEED"
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat
  --program-transport "$TRANSPORT" --family-strengths "$WEIGHTS"
  --projection-strength 1 --projection-min-t "$PMIN" --projection-max-t "$PMAX"
  --projection-interval-space edge_log_sigma_quantile --projection-schedule "$SCHEDULE"
  --projection-route-mode "$ROUTE" --projection-stability-mode rank
  --projection-stability-threshold .001 --projection-persistence-steps "$PERSIST"
  --projection-trust-radius "$TRUST" --guidance-stage both
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
  --seed "$SEED" --skip-detection --output "$OUT/eval_fast"
