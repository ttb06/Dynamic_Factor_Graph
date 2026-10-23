#!/usr/bin/env bash
# Test whether early stable-but-unreadable states and action geometry explain
# the full-graph fidelity loss. No LLM or API is used online.
#SBATCH --job-name=llmtd-news-valley-transport
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

LABEL=""; ROUTE=auto_edge_valley_isotonic; TRANSPORT=rank; FAMILY_TRANSPORTS=""
PERSIST=1; TRUST=0; PMIN=0; PMAX=.15; SCHEDULE=flat
WEIGHTS=llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5

case "$i" in
  0) LABEL=full_rank_edge_isotonic; ROUTE=auto_edge_isotonic;;
  1) LABEL=full_rank_valley;;
  2) LABEL=hybrid_order_rank_valley; TRANSPORT=direct; FAMILY_TRANSPORTS=statistical_order_program=rank;;
  3) LABEL=hybrid_algebraic_rank_valley; TRANSPORT=direct;
     FAMILY_TRANSPORTS=statistical_sum_program=rank,statistical_conditional_sum_program=rank,statistical_functional_program=rank;;
  4) LABEL=full_rank_valley_persistence2; PERSIST=2;;
  5) LABEL=full_rank_valley_persistence2_trust05; PERSIST=2; TRUST=.50;;
  6) LABEL=full_rank_valley_persistence3_trust025; PERSIST=3; TRUST=.25;;
  7) LABEL=full_rank_valley_middle; PMIN=.10; PMAX=.40; SCHEDULE=middle_peak;;
  8) LABEL=hybrid_order_rank_valley_middle_trust; TRANSPORT=direct;
     FAMILY_TRANSPORTS=statistical_order_program=rank; PMIN=.10; PMAX=.40;
     SCHEDULE=middle_peak; PERSIST=2; TRUST=.50;;
  9) LABEL=algebraic_rank_valley;
     WEIGHTS=llm_directional_bound=0,statistical_order_program=0,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5;;
  10) LABEL=full_rank_valley_without_llm_edges;
      WEIGHTS=llm_directional_bound=0,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5;;
  11) LABEL=llm_directional_only_valley;
      WEIGHTS=llm_directional_bound=1,statistical_order_program=0,statistical_functional_program=0,statistical_sum_program=0,statistical_conditional_sum_program=0;;
esac

SEED=20261021
OUT="$A/outputs/news_valley_transport_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
ARGS=(
  --repo "$REPO" --dataset news --regression-target-mode numerical
  --graph "$GRAPH" --config "$CONFIG" --checkpoint "$CKPT"
  --samples 4096 --steps 50 --seed "$SEED"
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
