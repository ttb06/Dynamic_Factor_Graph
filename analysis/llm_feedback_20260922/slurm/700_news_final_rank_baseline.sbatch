#!/usr/bin/env bash
# Matched final-only baseline for order rank transport (same graph/operator).
#SBATCH --job-name=llmtd-news-final-rank
#SBATCH --partition=cpu
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
i=$SLURM_ARRAY_TASK_ID

case "$i" in
  0) LABEL=s10_p1;  STRENGTH=.10; PASSES=1;  TRAJ="$A/outputs/news_order_rank_1091204_0_s10_late25";;
  1) LABEL=s10_p11; STRENGTH=.10; PASSES=11; TRAJ="$A/outputs/news_order_rank_1091204_0_s10_late25";;
  2) LABEL=s15_p1;  STRENGTH=.15; PASSES=1;  TRAJ="$A/outputs/news_order_rank_1091204_1_s15_late25";;
  3) LABEL=s15_p11; STRENGTH=.15; PASSES=11; TRAJ="$A/outputs/news_order_rank_1091204_1_s15_late25";;
  4) LABEL=s25_p1;  STRENGTH=.25; PASSES=1;  TRAJ="$A/outputs/news_rank_transport_1091168_4_order_s25";;
  5) LABEL=s25_p11; STRENGTH=.25; PASSES=11; TRAJ="$A/outputs/news_rank_transport_1091168_4_order_s25";;
esac

OUT="$A/outputs/news_final_rank_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
mkdir -p "$OUT"
cd "$REPO"
"$PY" "$A/scripts/apply_final_rank_transport.py" \
  --sample "$TRAJ/samples/unconditional.csv" \
  --graph "$A/outputs/news_documented_semantic_graph_20261016/graph.json" \
  --strength "$STRENGTH" --passes "$PASSES" \
  --output "$OUT/final_rank.csv"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$TRAJ/samples/unconditional.csv" \
  --sample "final_rank=$OUT/final_rank.csv" \
  --sample "trajectory_rank=$TRAJ/samples/dense_factor.csv" \
  --seed 20261604 --output "$OUT/eval"
