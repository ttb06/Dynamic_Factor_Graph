#!/usr/bin/env bash
# Family-weighted documented graph with train-only conditional-density veto.
#SBATCH --job-name=llmtd-news-density-veto
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
i=$SLURM_ARRAY_TASK_ID

VETO=()
case "$i" in
  0) LABEL=weighted;       SLACK=0;    WEIGHTS='documented_summary_order=.25,documented_functional_dependency=.25,documented_simplex=.05,documented_conditional_simplex=.05';;
  1) LABEL=veto_s0;        SLACK=0;    VETO=(--projection-density-veto); WEIGHTS='documented_summary_order=.25,documented_functional_dependency=.25,documented_simplex=.05,documented_conditional_simplex=.05';;
  2) LABEL=veto_s005;      SLACK=.05;  VETO=(--projection-density-veto); WEIGHTS='documented_summary_order=.25,documented_functional_dependency=.25,documented_simplex=.05,documented_conditional_simplex=.05';;
  3) LABEL=no_simplex;     SLACK=.02;  VETO=(--projection-density-veto); WEIGHTS='documented_summary_order=.25,documented_functional_dependency=.25,documented_simplex=0,documented_conditional_simplex=0';;
  4) LABEL=weak_simplex;   SLACK=.02;  VETO=(--projection-density-veto); WEIGHTS='documented_summary_order=.25,documented_functional_dependency=.15,documented_simplex=.03,documented_conditional_simplex=.03';;
  5) LABEL=balanced_weak;  SLACK=.05;  VETO=(--projection-density-veto); WEIGHTS='documented_summary_order=.15,documented_functional_dependency=.15,documented_simplex=.05,documented_conditional_simplex=.05';;
esac

OUT="$A/outputs/news_density_veto_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news \
  --graph "$A/outputs/news_documented_semantic_graph_20261016/graph.json" \
  --family-strengths "$WEIGHTS" \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261604 \
  --mode residual --strength 0 --passes 1 --trees 96 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --projection-strength 1 --projection-min-t 0 --projection-max-t .25 \
  --projection-interval-space log_sigma_quantile --projection-schedule flat \
  "${VETO[@]}" --projection-density-slack "$SLACK" \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "documented_final_projection=$OUT/samples/final_repair.csv" \
  --sample "weighted_trajectory=$OUT/samples/dense_factor.csv" \
  --seed 20261604 --output "$OUT/eval"
