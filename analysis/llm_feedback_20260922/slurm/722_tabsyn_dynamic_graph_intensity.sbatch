#!/usr/bin/env bash
# Focused intensity/activation sweep after the first TabSyn port showed small
# but jointly positive Shape/Trend/AUC changes at middle strength .20.
#SBATCH --job-name=llmtd-tabsyn-dyngraph2
#SBATCH --partition=prioritized
#SBATCH --array=0-3%4
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

PERSIST=--require-persistence
case "$i" in
  0) LABEL=middle_s040_r05; STRENGTH=.40; TRUST=.5;;
  1) LABEL=middle_s080_r05; STRENGTH=.80; TRUST=.5;;
  2) LABEL=middle_s020_nopersist; STRENGTH=.20; TRUST=.5; PERSIST=;;
  3) LABEL=middle_s040_nopersist; STRENGTH=.40; TRUST=.5; PERSIST=;;
esac

OUT="$A/outputs/tabsyn_dynamic_graph2_shoppers_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/tabsyn_dynamic_graph_guidance.py" \
  --tabsyn-repo "$TABSYN" --dataset shoppers --samples 4096 --steps 50 \
  --seed 20261017 --strength "$STRENGTH" --q-min .20 --q-max .55 \
  --trust-radius "$TRUST" $PERSIST --output "$OUT"

"$ROOT/baseline_runs/envs/grdm/bin/python" \
  "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --sample "tabsyn=$OUT/tabsyn_baseline.csv" \
  --sample "tabsyn_dynamic_graph=$OUT/tabsyn_dynamic_graph.csv" \
  --seed 20261017 --max-rows 4096 --output "$OUT/eval"
