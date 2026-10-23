#!/usr/bin/env bash
# Fixed-budget continuation frontier for the official TabSyn checkpoint.  Each
# checkpoint is immediately evaluated both unmodified and with the same
# middle-step dynamic graph add-on.
#SBATCH --job-name=llmtd-tabsyn-ft-graph
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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
  0) LABEL=e250_lr1e4; EPOCHS=250; LR=.0001;;
  1) LABEL=e750_lr1e4; EPOCHS=750; LR=.0001;;
  2) LABEL=e250_lr3e4; EPOCHS=250; LR=.0003;;
  3) LABEL=e750_lr3e4; EPOCHS=750; LR=.0003;;
esac
OUT="$A/outputs/tabsyn_shoppers_finetune_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"

cd "$REPO"
"$PY" "$A/scripts/finetune_tabsyn_diffusion.py" \
  --tabsyn-repo "$TABSYN" --dataset shoppers \
  --base-checkpoint "$TABSYN/tabsyn/ckpt/shoppers/model.pt" \
  --epochs "$EPOCHS" --learning-rate "$LR" --seed 20261018 --output "$OUT"

"$PY" "$A/scripts/tabsyn_dynamic_graph_guidance.py" \
  --tabsyn-repo "$TABSYN" --dataset shoppers --checkpoint "$OUT/model_best.pt" \
  --samples 9864 --steps 50 --seed 20261018 --strength .20 --trust-radius .50 \
  --route-mode fixed --q-min .20 --q-max .55 --require-persistence \
  --guidance-stage both --output "$OUT/sampling"

"$EVALPY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --sample "tabsyn_finetuned=$OUT/sampling/tabsyn_baseline.csv" \
  --sample "tabsyn_finetuned_dynamic=$OUT/sampling/tabsyn_dynamic_graph.csv" \
  --seed 20261018 --output "$OUT/eval"
