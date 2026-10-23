#!/usr/bin/env bash
# Correct exact endpoint comparator for the LCATS-style domain experiment.
#SBATCH --job-name=llmtd-news-domain-final
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SOURCE="$A/outputs/news_lcats_domain_trajectory_1101671/control/samples/unconditional.csv"
GRAPH="$A/outputs/news_lcats_domain_graph_20261020/combined_graph.json"
OUT="$A/outputs/news_lcats_exact_final_${SLURM_JOB_ID}"
mkdir -p "$OUT"

"$PY" "$A/scripts/lcats_domain_graph.py" repair \
  --graph "$GRAPH" --input "$SOURCE" --output "$OUT/exact_final.csv"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "Backbone=$SOURCE" \
  --sample "ExactDomainFinal=$OUT/exact_final.csv" \
  --seed 20261230 --skip-detection --output "$OUT/eval"

"$PY" "$A/scripts/lcats_domain_graph.py" audit \
  --graph "$GRAPH" \
  --sample "Backbone=$SOURCE" \
  --sample "ExactDomainFinal=$OUT/exact_final.csv" \
  --output "$OUT/domain_audit"
