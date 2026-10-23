#!/usr/bin/env bash
# Manifold-preserving ablation inspired by DSG / Trust Sampling.  The typed
# graph projection is capped in learned feature-sigma units before it changes
# the clean estimate used by the reverse update.
#SBATCH --job-name=llmtd-news-trust-proj
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-4%5
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
case "$i" in
  0) LABEL=clean_r025; LO=0; HI=.25; R=.25;;
  1) LABEL=clean_r050; LO=0; HI=.25; R=.50;;
  2) LABEL=clean_r100; LO=0; HI=.25; R=1.00;;
  3) LABEL=clean_r200; LO=0; HI=.25; R=2.00;;
  4) LABEL=full_r100;  LO=0; HI=1;   R=1.00;;
esac

OUT="$A/outputs/news_trust_projection_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news \
  --graph "$A/outputs/news_public_typed_graph_20261015/graph.json" \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261603 \
  --mode copula --strength 0 --passes 1 \
  --trees 192 --min-r2 .02 --clip 2.0 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --projection-strength 1 --projection-interval-space log_sigma_quantile \
  --projection-min-t "$LO" --projection-max-t "$HI" \
  --projection-schedule flat --projection-trust-radius "$R" \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "final_repair=$OUT/samples/final_repair.csv" \
  --sample "trust_projection=$OUT/samples/dense_factor.csv" \
  --sample "trust_then_certifier=$OUT/samples/dense_factor_then_certifier.csv" \
  --seed 20261603 --output "$OUT/eval"
