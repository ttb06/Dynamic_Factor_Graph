#!/usr/bin/env bash
# Main-metric experiment on externally documented News measurement programs.
# LLM/statistical hypotheses are not treated as validity labels in this job.
#SBATCH --job-name=llmtd-news-doc-hyper
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

MODE=copula
FACTOR=0
PASSES=1
FLO=0
FHI=0
PSTRENGTH=1
case "$i" in
  0) LABEL=clean_exact;       PLO=0;   PHI=.25;;
  1) LABEL=full_exact;        PLO=0;   PHI=1;;
  2) LABEL=middle_exact;      PLO=.20; PHI=.60;;
  3) LABEL=clean_half;        PLO=0;   PHI=.25; PSTRENGTH=.5;;
  4) LABEL=middle_residual;   PLO=0;   PHI=.25; PSTRENGTH=.5; MODE=residual; FACTOR=.03; FLO=.20; FHI=.60;;
  5) LABEL=middle_copula;     PLO=0;   PHI=.25; PSTRENGTH=.5; MODE=copula;   FACTOR=.03; FLO=.20; FHI=.60;;
esac

OUT="$A/outputs/news_documented_hypergraph_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news \
  --graph "$A/outputs/news_documented_semantic_graph_20261016/graph.json" \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261604 \
  --mode "$MODE" --strength "$FACTOR" --passes "$PASSES" \
  --trees 192 --min-r2 .05 --clip 1.0 \
  --interval-space log_sigma_quantile --factor-min-t "$FLO" --factor-max-t "$FHI" \
  --factor-schedule flat \
  --projection-strength "$PSTRENGTH" \
  --projection-interval-space log_sigma_quantile \
  --projection-min-t "$PLO" --projection-max-t "$PHI" \
  --projection-schedule flat \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "documented_final_projection=$OUT/samples/final_repair.csv" \
  --sample "documented_trajectory=$OUT/samples/dense_factor.csv" \
  --sample "trajectory_then_documented_certifier=$OUT/samples/dense_factor_then_certifier.csv" \
  --seed 20261604 --output "$OUT/eval"
