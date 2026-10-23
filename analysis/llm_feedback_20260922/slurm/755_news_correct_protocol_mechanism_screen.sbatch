#!/usr/bin/env bash
# One-seed, twelve-hypothesis screen on the corrected News representation.
# LLMs are absent online: every route/action is computed from the typed graph,
# train-fitted statistics, and the current denoising trajectory.
#SBATCH --job-name=llmtd-news-mech-screen
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

# Shared defaults: released stochastic-Heun sampler, automatic trajectory
# routing, and late actuation.  Each case changes one scientifically named
# component or graph scope.
LABEL=""; FAMILIES="llm_directional_bound"; FAMILY_WEIGHTS=""
STRENGTH=1; PMIN=0; PMAX=.15; INTERVAL=log_sigma_quantile; SCHEDULE=flat
ROUTE=auto_isotonic; STAGE=predictor; PERSIST=1; TRUST=0; TREES=0
TRANSPORT=direct; DENSITY_VETO=0; DENSITY_SLACK=0

case "$i" in
  0) LABEL=directional_predictor_late;;
  1) LABEL=directional_both_late; STAGE=both;;
  2) LABEL=directional_corrector_late; STAGE=corrector;;
  3) LABEL=directional_both_middle; STAGE=both; PMIN=.10; PMAX=.40; SCHEDULE=middle_peak;;
  4) LABEL=directional_edge_noise; STAGE=both; INTERVAL=edge_log_sigma_quantile; ROUTE=auto_edge_isotonic;;
  5) LABEL=directional_persistence3; STAGE=both; PERSIST=3;;
  6) LABEL=directional_per_edge_route; STAGE=both; ROUTE=auto_edge_isotonic;;
  7) LABEL=algebraic_per_edge; STAGE=both; STRENGTH=.50; ROUTE=auto_edge_isotonic;
     INTERVAL=edge_log_sigma_quantile;
     FAMILIES=statistical_sum_program,statistical_conditional_sum_program,statistical_functional_program;;
  8) LABEL=order_rank_per_edge; STAGE=both; STRENGTH=.25; ROUTE=auto_edge_isotonic;
     INTERVAL=edge_log_sigma_quantile; TRANSPORT=rank; FAMILIES=statistical_order_program;;
  9) LABEL=full_weighted_per_edge; STAGE=both; ROUTE=auto_edge_isotonic;
     INTERVAL=edge_log_sigma_quantile; FAMILIES="";
     FAMILY_WEIGHTS=llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5;;
  10) LABEL=full_density_veto; STAGE=both; ROUTE=auto_edge_isotonic;
      INTERVAL=edge_log_sigma_quantile; FAMILIES=""; TREES=64; DENSITY_VETO=1; DENSITY_SLACK=.05;
      FAMILY_WEIGHTS=llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5;;
  11) LABEL=full_persistent_middle_trust; STAGE=both; ROUTE=auto_edge_isotonic;
      PMIN=.10; PMAX=.40; SCHEDULE=middle_peak; PERSIST=2; TRUST=.50; FAMILIES="";
      FAMILY_WEIGHTS=llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5;;
esac

SEED=20261019
OUT="$A/outputs/news_correct_mechanism_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
ARGS=(
  --repo "$REPO" --dataset news --regression-target-mode numerical
  --graph "$GRAPH" --config "$CONFIG" --checkpoint "$CKPT"
  --samples 4096 --steps 50 --seed "$SEED"
  --mode copula --strength 0 --passes 1 --trees "$TREES" --min-r2 .05 --clip 1
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat
  --program-transport "$TRANSPORT"
  --projection-strength "$STRENGTH" --projection-min-t "$PMIN" --projection-max-t "$PMAX"
  --projection-interval-space "$INTERVAL" --projection-schedule "$SCHEDULE"
  --projection-route-mode "$ROUTE" --projection-stability-mode rank
  --projection-stability-threshold .001 --projection-persistence-steps "$PERSIST"
  --projection-trust-radius "$TRUST" --guidance-stage "$STAGE"
  --stochastic-sampler --second-order-correction --output "$OUT"
)
if [[ -n "$FAMILIES" ]]; then ARGS+=(--include-families "$FAMILIES"); fi
if [[ -n "$FAMILY_WEIGHTS" ]]; then ARGS+=(--family-strengths "$FAMILY_WEIGHTS"); fi
if (( DENSITY_VETO == 1 )); then
  ARGS+=(--projection-density-veto --projection-density-slack "$DENSITY_SLACK")
fi

cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" "${ARGS[@]}"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "matched_final_repair=$OUT/samples/final_repair.csv" \
  --sample "dynamic=$OUT/samples/dense_factor.csv" \
  --sample "dynamic_then_certifier=$OUT/samples/dense_factor_then_certifier.csv" \
  --seed "$SEED" --skip-detection --output "$OUT/eval_fast"
