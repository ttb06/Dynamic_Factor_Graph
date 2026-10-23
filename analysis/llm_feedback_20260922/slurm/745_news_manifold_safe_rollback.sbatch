#!/usr/bin/env bash
# Train-only manifold rollback over paired, already-certified News branches.
# Six limited-interval candidates x two conservative support tolerances.
#SBATCH --job-name=llmtd-news-rollback
#SBATCH --partition=prioritized
#SBATCH --array=0-11%6
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
i=$SLURM_ARRAY_TASK_ID
source_index=$((i % 6))
slack_index=$((i / 6))
case "$source_index" in
  0) SOURCE="$A/outputs/news_limited_interval_1093945_0_w05_20_s075"; LABEL=w05_20_s075;;
  1) SOURCE="$A/outputs/news_limited_interval_1093945_1_w05_20_s100"; LABEL=w05_20_s100;;
  2) SOURCE="$A/outputs/news_limited_interval_1093945_2_w10_25_s075"; LABEL=w10_25_s075;;
  3) SOURCE="$A/outputs/news_limited_interval_1093945_3_w10_25_s100"; LABEL=w10_25_s100;;
  4) SOURCE="$A/outputs/news_limited_interval_1093945_4_w15_30_s075"; LABEL=w15_30_s075;;
  5) SOURCE="$A/outputs/news_limited_interval_1093945_5_w15_30_s100"; LABEL=w15_30_s100;;
esac
if [[ "$slack_index" == 0 ]]; then SLACK=0; SLABEL=strict; else SLACK=.002; SLABEL=slack002; fi
OUT="$A/outputs/news_manifold_rollback_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}_${SLABEL}"
mkdir -p "$OUT/samples"

cd "$REPO"
"$PY" "$A/scripts/paired_manifold_rollback.py" \
  --train "$REPO/synthetic/news/real.csv" \
  --reference "$SOURCE/samples/final_repair.csv" \
  --candidate "$SOURCE/samples/dense_factor_then_certifier.csv" \
  --output "$OUT/samples/manifold_rollback.csv" \
  --prototypes 256 --components 20 --relative-slack "$SLACK" --seed 20261607

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$SOURCE/samples/unconditional.csv" \
  --sample "final_repair=$SOURCE/samples/final_repair.csv" \
  --sample "limited_then_certifier=$SOURCE/samples/dense_factor_then_certifier.csv" \
  --sample "manifold_rollback=$OUT/samples/manifold_rollback.csv" \
  --seed 20261607 --output "$OUT/eval"
