#!/usr/bin/env bash
# Matched graph-source ablation under the same per-edge automatic trajectory
# controller.  This separates LLM discovery value from routing/actuation value.
#SBATCH --job-name=llmtd-news-graph-source
#SBATCH --partition=prioritized
#SBATCH --array=0-5%6
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
i=$SLURM_ARRAY_TASK_ID

case "$i" in
  0) LABEL=documented; FAMILIES=statistical_order_program; WEIGHTS=statistical_order_program=.25; SEED=20261605;;
  1) LABEL=documented; FAMILIES=statistical_order_program; WEIGHTS=statistical_order_program=.25; SEED=20261606;;
  2) LABEL=documented; FAMILIES=statistical_order_program; WEIGHTS=statistical_order_program=.25; SEED=20261607;;
  3) LABEL=union; FAMILIES=statistical_order_program,llm_directional_bound; WEIGHTS=statistical_order_program=.25,llm_directional_bound=.75; SEED=20261605;;
  4) LABEL=union; FAMILIES=statistical_order_program,llm_directional_bound; WEIGHTS=statistical_order_program=.25,llm_directional_bound=.75; SEED=20261606;;
  5) LABEL=union; FAMILIES=statistical_order_program,llm_directional_bound; WEIGHTS=statistical_order_program=.25,llm_directional_bound=.75; SEED=20261607;;
esac

OUT="$A/outputs/news_graph_source_auto_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}_s${SEED}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news --graph "$GRAPH" \
  --include-families "$FAMILIES" --family-strengths "$WEIGHTS" \
  --program-transport rank \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed "$SEED" \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --interval-space log_sigma_quantile \
  --projection-strength 1 --projection-min-t 0 --projection-max-t .15 \
  --projection-interval-space log_sigma_quantile --projection-schedule flat \
  --projection-route-mode auto_edge_isotonic \
  --projection-stability-mode rank --projection-stability-threshold .001 \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "final_repair=$OUT/samples/final_repair.csv" \
  --sample "$LABEL=$OUT/samples/dense_factor.csv" \
  --sample "${LABEL}_then_certifier=$OUT/samples/dense_factor_then_certifier.csv" \
  --seed "$SEED" --output "$OUT/eval"
