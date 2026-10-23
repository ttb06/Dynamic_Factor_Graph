#!/usr/bin/env bash
# End-to-end action selection: distill only graph moves that reduce train x0 error.
#SBATCH --job-name=llmtd-news-e2e-benefit
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:a40:1
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
TRAIN="$A/outputs/full_train_news_official_regression_1093974"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
i=$SLURM_ARRAY_TASK_ID
NAME=unknown; FILTER=cell; AUX=.03; VIEWS=1; TRAJECTORY=0; TMIN=.10; TMAX=.40; EXTRA=()
case "$i" in
  0) NAME=cell_w03;;
  1) NAME=row_w03; FILTER=row;;
  2) NAME=cell_w10; AUX=.10;;
  3) NAME=cell_broad_dual; VIEWS=2; TRAJECTORY=.10; TMIN=.05; TMAX=.65; EXTRA+=(--stable-only);;
esac
OUT="$A/outputs/news_e2e_benefit_${NAME}_${SLURM_ARRAY_JOB_ID}_${i}"
cd "$REPO"
"$PY" "$A/scripts/finetune_trajectory_teacher_tabdiff.py" \
  --repo "$REPO" --dataset news --regression-target-mode numerical \
  --config "$TRAIN/config.pkl" --checkpoint "$TRAIN/ema_model_epoch_4000.pt" \
  --graph "$GRAPH" --output "$OUT/train" --epochs 50 --batch-size 2048 \
  --learning-rate 3e-6 --trainable-scope full --target-mode sparse \
  --teacher-transport direct --teacher-strength .5 --benefit-filter "$FILTER" \
  --auxiliary-weight "$AUX" --anchor-weight 1 \
  --views "$VIEWS" --trajectory-weight "$TRAJECTORY" \
  --t-min "$TMIN" --t-max "$TMAX" --seed 20261025 "${EXTRA[@]}"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news --regression-target-mode numerical \
  --graph "$GRAPH" --config "$TRAIN/config.pkl" \
  --checkpoint "$OUT/train/ema_model_final.pt" \
  --samples 8192 --steps 50 --seed 20261024 \
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
  --seed 20261024 --skip-detection --output "$OUT/eval"

