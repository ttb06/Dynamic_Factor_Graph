#!/usr/bin/env bash
# Paper-inspired ablation of when dense graph guidance is applied.
# Dou et al. (NeurIPS 2024) found that CFG is most useful in a bounded
# intermediate-noise interval.  We keep the graph, seed, strength and sampler
# fixed, and change only the active interval/schedule.
#SBATCH --job-name=llmtd-news-noise-window
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

# log-sigma quantile: 0 is the clean end, 1 is the noisiest end.
case "$i" in
  0) LABEL=full_flat;       LO=0;   HI=1;   SCHEDULE=flat;;
  1) LABEL=middle_flat;     LO=.20; HI=.60; SCHEDULE=flat;;
  2) LABEL=middle_smooth;   LO=.20; HI=.60; SCHEDULE=middle_peak;;
  3) LABEL=middle_narrow;   LO=.30; HI=.55; SCHEDULE=flat;;
  4) LABEL=clean_end_only;  LO=0;   HI=.25; SCHEDULE=flat;;
  5) LABEL=noisy_end_only;  LO=.65; HI=1;   SCHEDULE=flat;;
esac

OUT="$A/outputs/news_noise_window_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news \
  --graph "$A/outputs/news_public_typed_graph_20261015/graph.json" \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261602 \
  --mode copula --strength .35 --passes 3 \
  --trees 192 --min-r2 .02 --clip 2.0 \
  --interval-space log_sigma_quantile \
  --factor-min-t "$LO" --factor-max-t "$HI" --factor-schedule "$SCHEDULE" \
  --projection-strength 0 --projection-max-t 0 \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "final_repair=$OUT/samples/final_repair.csv" \
  --sample "dense_factor=$OUT/samples/dense_factor.csv" \
  --sample "dense_factor_then_certifier=$OUT/samples/dense_factor_then_certifier.csv" \
  --seed 20261602 --output "$OUT/eval"
