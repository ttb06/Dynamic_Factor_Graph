#!/usr/bin/env bash
#SBATCH --job-name=llmtd-transition-survival
#SBATCH --partition=prioritized
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=10:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(shoppers shoppers shoppers default default default diabetes diabetes diabetes)
SEEDS=(20261061 20261062 20261063 20261061 20261062 20261063 20261061 20261062 20261063)
i=$SLURM_ARRAY_TASK_ID
dataset=${DATASETS[$i]}
seed=${SEEDS[$i]}

if [[ "$dataset" == shoppers ]]; then
  run=full_train_shoppers_1033761
  catalog=expanded_constraint_catalog_1057680
  samples=8192
elif [[ "$dataset" == default ]]; then
  run=full_train_default_1033760
  catalog=cross_dataset_catalog_default_1058707_1
  samples=8192
else
  run=full_train_diabetes_1048479
  catalog=diabetes_catalog_1060368
  samples=4096
fi

cd "$REPO"
"$PY" "$A/scripts/transition_survival_probe.py" \
  --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/$catalog/constraint_catalog.json" \
  --config "$A/outputs/$run/config.pkl" \
  --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
  --samples "$samples" --steps 50 --seed "$seed" --max-wait 5 \
  --active-min-t .10 --active-max-t .75 \
  --output "$A/outputs/transition_survival_${dataset}_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
