#!/usr/bin/env bash
# Separate sampling variance from training variance using fixed checkpoints.
#SBATCH --job-name=llmtd-news-e2e-resample
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-7%8
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
TRAIN="$A/outputs/full_train_news_official_regression_1093974"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
CONTINUE="$A/outputs/news_e2e_teacher_continue_full_1094488_0/train/ema_model_final.pt"
SPARSE="$A/outputs/news_e2e_teacher_sparse_full_1094488_2/train/ema_model_final.pt"
BROAD="$A/outputs/news_e2e_teacher_broad_dual_1094488_9/train/ema_model_final.pt"
i=$SLURM_ARRAY_TASK_ID
case "$i" in
  0) NAME=continue_s26; CKPT="$CONTINUE"; SEED=20261026;;
  1) NAME=sparse_s26; CKPT="$SPARSE"; SEED=20261026;;
  2) NAME=continue_s27; CKPT="$CONTINUE"; SEED=20261027;;
  3) NAME=sparse_s27; CKPT="$SPARSE"; SEED=20261027;;
  4) NAME=continue_s28; CKPT="$CONTINUE"; SEED=20261028;;
  5) NAME=sparse_s28; CKPT="$SPARSE"; SEED=20261028;;
  6) NAME=broad_s26; CKPT="$BROAD"; SEED=20261026;;
  7) NAME=broad_s27; CKPT="$BROAD"; SEED=20261027;;
esac
OUT="$A/outputs/news_e2e_resample_${NAME}_${SLURM_ARRAY_JOB_ID}_${i}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news --regression-target-mode numerical \
  --graph "$GRAPH" --config "$TRAIN/config.pkl" --checkpoint "$CKPT" \
  --samples 8192 --steps 50 --seed "$SEED" \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --family-strengths llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5 \
  --family-transports statistical_order_program=rank \
  --projection-strength 1 --projection-min-t .10 --projection-max-t .40 \
  --projection-interval-space edge_log_sigma_quantile --projection-schedule middle_peak \
  --projection-route-mode auto_edge_valley_isotonic --projection-stability-mode rank \
  --projection-stability-threshold .001 --projection-persistence-steps 2 \
  --projection-trust-radius .50 --guidance-stage both \
  --stochastic-sampler --second-order-correction --output "$OUT/sample"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "e2e_${NAME}=$OUT/sample/samples/unconditional.csv" \
  --sample "e2e_${NAME}_dynamic=$OUT/sample/samples/dense_factor.csv" \
  --seed "$SEED" --skip-detection --output "$OUT/eval"

