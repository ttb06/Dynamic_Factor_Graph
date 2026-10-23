#!/usr/bin/env bash
# One-seed screen for positive graph messages on the corrected News protocol.
# The LLM is strictly offline: online messages are train-fitted ExtraTrees
# potentials, typed graph programs, trajectory state, and deterministic gates.
#SBATCH --job-name=llmtd-news-positive-screen
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

LABEL=""; MODE=copula; STRENGTH=.25; PASSES=1; CLIP=1
FMIN=.10; FMAX=.55; FSCHEDULE=middle_peak; RESIDUAL_GATE=0; UNCERTAINTY_GATE=-1
SAT_SCALE=1; VIOL_SCALE=1; FACTOR_FAMILIES=""
PROJ_STRENGTH=0; PMIN=0; PMAX=.15; PSCHEDULE=flat; ROUTE=auto_edge_isotonic
PERSIST=1; TRUST=0; FAMILY_WEIGHTS=""; TRANSPORT=direct

case "$i" in
  0) LABEL=positive_all_s010; STRENGTH=.10;;
  1) LABEL=positive_all_s025; STRENGTH=.25;;
  2) LABEL=positive_all_s050; STRENGTH=.50;;
  3) LABEL=positive_residual05_uncertainty1; RESIDUAL_GATE=.50; UNCERTAINTY_GATE=1;;
  4) LABEL=positive_residual10_uncertainty1; RESIDUAL_GATE=1; UNCERTAINTY_GATE=1;;
  5) LABEL=positive_satisfied_only; SAT_SCALE=1; VIOL_SCALE=0;;
  6) LABEL=positive_violating_only; SAT_SCALE=0; VIOL_SCALE=1;;
  7) LABEL=residual_transport_s010; MODE=residual; STRENGTH=.10;;
  8) LABEL=positive_order_factors; FACTOR_FAMILIES=statistical_order_program;;
  9) LABEL=positive_algebraic_factors; FACTOR_FAMILIES=statistical_sum_program,statistical_conditional_sum_program,statistical_functional_program;;
  10) LABEL=positive_plus_order_rank; PROJ_STRENGTH=.25; TRANSPORT=rank;
      FAMILY_WEIGHTS=statistical_order_program=1,llm_directional_bound=0,statistical_functional_program=0,statistical_sum_program=0,statistical_conditional_sum_program=0;;
  11) LABEL=positive_plus_full_persistent; PROJ_STRENGTH=1; PMIN=.10; PMAX=.40;
      PSCHEDULE=middle_peak; PERSIST=2; TRUST=.50;
      FAMILY_WEIGHTS=llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5;;
esac

SEED=20261020
OUT="$A/outputs/news_positive_graph_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
ARGS=(
  --repo "$REPO" --dataset news --regression-target-mode numerical
  --graph "$GRAPH" --config "$CONFIG" --checkpoint "$CKPT"
  --samples 4096 --steps 50 --seed "$SEED"
  --mode "$MODE" --strength "$STRENGTH" --passes "$PASSES" --trees 128 --min-r2 .05 --clip "$CLIP"
  --factor-min-t "$FMIN" --factor-max-t "$FMAX" --factor-schedule "$FSCHEDULE"
  --interval-space log_sigma_quantile --factor-residual-gate "$RESIDUAL_GATE"
  --factor-uncertainty-gate "$UNCERTAINTY_GATE" --factor-uncertainty-trees 16
  --factor-satisfied-strength-scale "$SAT_SCALE" --factor-violating-strength-scale "$VIOL_SCALE"
  --program-transport "$TRANSPORT"
  --projection-strength "$PROJ_STRENGTH" --projection-min-t "$PMIN" --projection-max-t "$PMAX"
  --projection-interval-space edge_log_sigma_quantile --projection-schedule "$PSCHEDULE"
  --projection-route-mode "$ROUTE" --projection-stability-mode rank
  --projection-stability-threshold .001 --projection-persistence-steps "$PERSIST"
  --projection-trust-radius "$TRUST" --guidance-stage both
  --stochastic-sampler --second-order-correction --output "$OUT"
)
if [[ -n "$FACTOR_FAMILIES" ]]; then ARGS+=(--factor-families "$FACTOR_FAMILIES"); fi
if [[ -n "$FAMILY_WEIGHTS" ]]; then ARGS+=(--family-strengths "$FAMILY_WEIGHTS"); fi

cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" "${ARGS[@]}"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "matched_final_repair=$OUT/samples/final_repair.csv" \
  --sample "dynamic=$OUT/samples/dense_factor.csv" \
  --sample "dynamic_then_certifier=$OUT/samples/dense_factor_then_certifier.csv" \
  --seed "$SEED" --skip-detection --output "$OUT/eval_fast"
