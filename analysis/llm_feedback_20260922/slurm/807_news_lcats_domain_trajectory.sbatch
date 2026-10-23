#!/usr/bin/env bash
# Same-node screening of LCATS Table-1 integer/domain-bound primitives as
# dynamic clean-preview actions, plus their exact endpoint-repair control.
#SBATCH --job-name=llmtd-news-lcats-domain
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
GRAPHROOT="$A/outputs/news_lcats_domain_graph_20261020"
BASEGRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
OUT="$A/outputs/news_lcats_domain_trajectory_${SLURM_JOB_ID}"
SEED=20261230

typed_weights="llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5"
zero_typed="llm_directional_bound=0,statistical_order_program=0,statistical_functional_program=0,statistical_sum_program=0,statistical_conditional_sum_program=0"

# Build/freeze the verified domain graphs on the allocated compute node.  This
# also prevents metadata regeneration from loading pandas/data on a login node.
mkdir -p "$GRAPHROOT"
for families_and_name in "integer integer_graph" "bounds bounds_graph" "integer,bounds combined_graph"; do
  read -r families name <<< "$families_and_name"
  "$PY" "$A/scripts/lcats_domain_graph.py" build \
    --dataset news --train "$REPO/data/news/train.csv" \
    --base-graph "$BASEGRAPH" --families "$families" \
    --output "$GRAPHROOT/${name}.json" >/dev/null
done

for variant in control integer_s010 integer_s025 bounds_only domain_only typed_domain full_domain_conditional; do
  graph="$BASEGRAPH"
  weights="$typed_weights"
  transport=(--distribution-transport-mode none)
  case "$variant" in
    control) ;;
    integer_s010)
      graph="$GRAPHROOT/integer_graph.json"
      weights="$zero_typed,integer_domain=.10"
      ;;
    integer_s025)
      graph="$GRAPHROOT/integer_graph.json"
      weights="$zero_typed,integer_domain=.25"
      ;;
    bounds_only)
      graph="$GRAPHROOT/bounds_graph.json"
      weights="$zero_typed,domain_bound=1"
      ;;
    domain_only)
      graph="$GRAPHROOT/combined_graph.json"
      weights="$zero_typed,integer_domain=.25,domain_bound=1"
      ;;
    typed_domain)
      graph="$GRAPHROOT/combined_graph.json"
      weights="$typed_weights,integer_domain=.25,domain_bound=1"
      ;;
    full_domain_conditional)
      graph="$GRAPHROOT/combined_graph.json"
      weights="$typed_weights,integer_domain=.25,domain_bound=1"
      transport=(--distribution-transport-mode copula \
        --distribution-transport-scope all \
        --distribution-transport-condition-target-quantiles 4 \
        --distribution-transport-strength .10 \
        --distribution-transport-min-t .20 \
        --distribution-transport-max-t .35 \
        --distribution-transport-once)
      ;;
  esac

  cd "$REPO"
  "$PY" "$A/scripts/dense_numeric_factor_messages.py" \
    --repo "$REPO" --dataset news --regression-target-mode numerical \
    --graph "$graph" \
    --config "$A/outputs/full_train_news_official_regression_1093974/config.pkl" \
    --checkpoint "$A/outputs/news_e2e_teacher_continue_full_1094488_0/train/ema_model_final.pt" \
    --samples 4096 --steps 50 --seed "$SEED" \
    --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
    --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
    --family-strengths "$weights" \
    --family-transports statistical_order_program=rank \
    --projection-strength 1 \
    --projection-min-t .05 --projection-max-t .35 \
    --projection-interval-space log_sigma_quantile \
    --projection-schedule middle_peak \
    --projection-stability-mode scale --projection-stability-threshold .01 \
    --projection-persistence-steps 2 --projection-trust-radius .75 \
    "${transport[@]}" \
    --categorical-guidance-mode none \
    --guidance-stage both --stochastic-sampler --second-order-correction \
    --output "$OUT/$variant"
done

sha256sum "$OUT"/*/samples/unconditional.csv
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "Backbone=$OUT/control/samples/unconditional.csv" \
  --sample "TypedControl=$OUT/control/samples/dense_factor.csv" \
  --sample "Integer010=$OUT/integer_s010/samples/dense_factor.csv" \
  --sample "Integer025=$OUT/integer_s025/samples/dense_factor.csv" \
  --sample "BoundsOnly=$OUT/bounds_only/samples/dense_factor.csv" \
  --sample "DomainOnly=$OUT/domain_only/samples/dense_factor.csv" \
  --sample "TypedDomain=$OUT/typed_domain/samples/dense_factor.csv" \
  --sample "FullDomainConditional=$OUT/full_domain_conditional/samples/dense_factor.csv" \
  --sample "DomainFinalRepair=$OUT/domain_only/samples/final_repair.csv" \
  --seed "$SEED" --skip-detection --output "$OUT/eval"

"$PY" "$A/scripts/lcats_domain_graph.py" audit \
  --graph "$GRAPHROOT/combined_graph.json" \
  --sample "Backbone=$OUT/control/samples/unconditional.csv" \
  --sample "TypedControl=$OUT/control/samples/dense_factor.csv" \
  --sample "Integer010=$OUT/integer_s010/samples/dense_factor.csv" \
  --sample "Integer025=$OUT/integer_s025/samples/dense_factor.csv" \
  --sample "BoundsOnly=$OUT/bounds_only/samples/dense_factor.csv" \
  --sample "DomainOnly=$OUT/domain_only/samples/dense_factor.csv" \
  --sample "TypedDomain=$OUT/typed_domain/samples/dense_factor.csv" \
  --sample "FullDomainConditional=$OUT/full_domain_conditional/samples/dense_factor.csv" \
  --sample "DomainFinalRepair=$OUT/domain_only/samples/final_repair.csv" \
  --output "$OUT/domain_audit"
