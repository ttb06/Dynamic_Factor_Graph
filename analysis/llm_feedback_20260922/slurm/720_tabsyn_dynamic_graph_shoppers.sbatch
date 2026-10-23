#!/usr/bin/env bash
# Cross-backbone portability: training-free dynamic graph guidance inside the
# official TabSyn latent reverse process on Shoppers.
#SBATCH --job-name=llmtd-tabsyn-dyngraph
#SBATCH --partition=prioritized
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
TABSYN="$ROOT/baseline_repos/tabsyn"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/lcats-official/bin/python"
i=$SLURM_ARRAY_TASK_ID

case "$i" in
  0) LABEL=middle_s005; STRENGTH=.05; QMIN=.20; QMAX=.55;;
  1) LABEL=middle_s010; STRENGTH=.10; QMIN=.20; QMAX=.55;;
  2) LABEL=middle_s020; STRENGTH=.20; QMIN=.20; QMAX=.55;;
  3) LABEL=late_s010;   STRENGTH=.10; QMIN=.00; QMAX=.20;;
esac

OUT="$A/outputs/tabsyn_dynamic_graph_shoppers_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/tabsyn_dynamic_graph_guidance.py" \
  --tabsyn-repo "$TABSYN" --dataset shoppers --samples 4096 --steps 50 \
  --seed 20261017 --strength "$STRENGTH" --q-min "$QMIN" --q-max "$QMAX" \
  --trust-radius .5 --require-persistence --output "$OUT"

"$ROOT/baseline_runs/envs/grdm/bin/python" \
  "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --sample "tabsyn=$OUT/tabsyn_baseline.csv" \
  --sample "tabsyn_dynamic_graph=$OUT/tabsyn_dynamic_graph.csv" \
  --seed 20261017 --max-rows 4096 --output "$OUT/eval"
