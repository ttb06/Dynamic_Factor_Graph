#!/usr/bin/env bash
# TabSyn transfer with a backbone-robust automatic route selector.  Ignore the
# false-stability prefix, anchor at the minimum-stability valley, then select
# the steepest monotone recovery of graph readability.
#SBATCH --job-name=llmtd-tabsyn-valley
#SBATCH --partition=prioritized
#SBATCH --array=0-2%3
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
  0) LABEL=valley001; THRESHOLD=.001;;
  1) LABEL=valley005; THRESHOLD=.005;;
  2) LABEL=valley010; THRESHOLD=.010;;
esac
OUT="$A/outputs/tabsyn_valley_route_shoppers_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"

cd "$REPO"
"$PY" "$A/scripts/tabsyn_dynamic_graph_guidance.py" \
  --tabsyn-repo "$TABSYN" --dataset shoppers --samples 4096 --steps 50 \
  --seed 20261017 --strength .20 --trust-radius .5 \
  --route-mode auto_valley --actuation-max-q .15 \
  --stability-threshold "$THRESHOLD" --output "$OUT"

"$ROOT/baseline_runs/envs/grdm/bin/python" \
  "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --sample "tabsyn=$OUT/tabsyn_baseline.csv" \
  --sample "tabsyn_valley_route=$OUT/tabsyn_dynamic_graph.csv" \
  --seed 20261017 --max-rows 4096 --output "$OUT/eval"
