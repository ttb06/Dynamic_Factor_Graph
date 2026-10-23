#!/usr/bin/env bash
# Matched untouched-checkpoint control for the continuation frontier.
#SBATCH --job-name=llmtd-tabsyn-original
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
TABSYN="$ROOT/baseline_repos/tabsyn"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/lcats-official/bin/python"
EVALPY="$ROOT/baseline_runs/envs/grdm/bin/python"
OUT="$A/outputs/tabsyn_shoppers_matched_original_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/tabsyn_dynamic_graph_guidance.py" \
  --tabsyn-repo "$TABSYN" --dataset shoppers \
  --checkpoint "$TABSYN/tabsyn/ckpt/shoppers/model.pt" \
  --samples 9864 --steps 50 --seed 20261018 --strength .20 --trust-radius .50 \
  --route-mode fixed --q-min .20 --q-max .55 --require-persistence \
  --guidance-stage both --output "$OUT"
"$EVALPY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --sample "tabsyn_original=$OUT/tabsyn_baseline.csv" \
  --sample "tabsyn_original_dynamic=$OUT/tabsyn_dynamic_graph.csv" \
  --seed 20261018 --output "$OUT/eval"

