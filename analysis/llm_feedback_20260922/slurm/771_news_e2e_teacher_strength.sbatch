#!/usr/bin/env bash
# Higher-signal follow-up: the light frontier contributes only ~0.2% of total
# loss at weight .05.  This array tests whether weak effect is under-training.
#SBATCH --job-name=llmtd-news-e2e-strength
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
NAME=unknown; TARGET=sparse; AUX=.5; ANCHOR=1; EXTRA=()
case "$i" in
  0) NAME=sparse_w05; AUX=.5;;
  1) NAME=sparse_w10; AUX=1; ANCHOR=2;;
  2) NAME=sparse_pcgrad_w10; AUX=1; ANCHOR=2; EXTRA+=(--gradient-mode pcgrad);;
  3) NAME=delta_w02; TARGET=delta_normalized; AUX=.2; ANCHOR=1;;
esac
OUT="$A/outputs/news_e2e_strength_${NAME}_${SLURM_ARRAY_JOB_ID}_${i}"
cd "$REPO"
"$PY" "$A/scripts/finetune_trajectory_teacher_tabdiff.py" \
  --repo "$REPO" --dataset news --regression-target-mode numerical \
  --config "$TRAIN/config.pkl" --checkpoint "$TRAIN/ema_model_epoch_4000.pt" \
  --graph "$GRAPH" --output "$OUT/train" --epochs 50 --batch-size 2048 \
  --learning-rate 1e-5 --trainable-scope head --target-mode "$TARGET" \
  --teacher-transport direct --teacher-strength 1 \
  --auxiliary-weight "$AUX" --anchor-weight "$ANCHOR" \
  --t-min .10 --t-max .40 --seed 20261025 "${EXTRA[@]}"
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
  --sample "e2e_${NAME}_final=$OUT/sample/samples/final_repair.csv" \
  --sample "e2e_${NAME}_dynamic=$OUT/sample/samples/dense_factor.csv" \
  --seed 20261024 --skip-detection --output "$OUT/eval"

