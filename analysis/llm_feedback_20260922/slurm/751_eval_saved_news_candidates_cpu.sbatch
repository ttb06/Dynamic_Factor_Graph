#!/usr/bin/env bash
# Finish standard evaluation of already-saved samples without occupying GPUs.
#SBATCH --job-name=llmtd-news-eval-saved
#SBATCH --partition=cpu
#SBATCH --array=0-6%7
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
i=$SLURM_ARRAY_TASK_ID
case "$i" in
  0) SOURCE="$A/outputs/news_limited_interval_1093945_0_w05_20_s075";;
  1) SOURCE="$A/outputs/news_limited_interval_1093945_1_w05_20_s100";;
  2) SOURCE="$A/outputs/news_limited_interval_1093945_2_w10_25_s075";;
  3) SOURCE="$A/outputs/news_limited_interval_1093945_3_w10_25_s100";;
  4) SOURCE="$A/outputs/news_limited_interval_1093945_4_w15_30_s075";;
  5) SOURCE="$A/outputs/news_limited_interval_1093945_5_w15_30_s100";;
  6) SOURCE="$A/outputs/news_limited_interval_1093945_0_w05_20_s075";
     ROLLBACK="$A/outputs/news_manifold_rollback_1093951_0_w05_20_s075_strict/samples/manifold_rollback.csv";;
esac
OUT="$SOURCE/eval_cpu_${SLURM_ARRAY_JOB_ID}_${i}"
cd "$REPO"
ARGS=(
  --repo "$REPO"
  --sample "tabdiff=$SOURCE/samples/unconditional.csv"
  --sample "final_repair=$SOURCE/samples/final_repair.csv"
  --sample "limited=$SOURCE/samples/dense_factor.csv"
  --sample "limited_then_certifier=$SOURCE/samples/dense_factor_then_certifier.csv"
  --seed 20261607 --output "$OUT"
)
if [[ "$i" == 6 ]]; then ARGS+=(--sample "manifold_rollback=$ROLLBACK"); fi
"$PY" "$A/scripts/evaluate_news_public_constraints.py" "${ARGS[@]}"
