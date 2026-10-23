#!/usr/bin/env bash
# Port the dual-stage middle-interval solver to the untouched official TabSyn
# Shoppers checkpoint.  Strength is the only varied quantity.
#SBATCH --job-name=llmtd-tabsyn-mid-solver
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
TABSYN="$ROOT/baseline_repos/tabsyn"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/lcats-official/bin/python"
EVALPY="$ROOT/baseline_runs/envs/grdm/bin/python"
i=$SLURM_ARRAY_TASK_ID
case "$i" in
  0) LABEL=predictor10; STRENGTH=.10; STAGE=predictor;;
  1) LABEL=both05; STRENGTH=.05; STAGE=both;;
  2) LABEL=both10; STRENGTH=.10; STAGE=both;;
  3) LABEL=both20; STRENGTH=.20; STAGE=both;;
esac
OUT="$A/outputs/tabsyn_middle_solver_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"

cd "$REPO"
"$PY" "$A/scripts/tabsyn_dynamic_graph_guidance.py" \
  --tabsyn-repo "$TABSYN" --dataset shoppers --samples 9864 --steps 50 \
  --seed 20261019 --strength "$STRENGTH" --trust-radius .50 \
  --route-mode fixed --q-min .20 --q-max .55 --require-persistence \
  --guidance-stage "$STAGE" --output "$OUT"
"$EVALPY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --sample "tabsyn=$OUT/tabsyn_baseline.csv" \
  --sample "dynamic=$OUT/tabsyn_dynamic_graph.csv" \
  --seed 20261019 --output "$OUT/eval"
