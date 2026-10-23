#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dia-budget20
#SBATCH --partition=prioritized
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
SEEDS=(20261041 20261042 20261043); seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}
cd "$REPO"
for topology in dynamic frozen_state shuffled; do
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset diabetes \
    --graph "$A/outputs/diabetes_catalog_1060368/constraint_catalog.json" \
    --schedule "$A/outputs/diabetes_trajectory_1060369/typed_readability_schedule.csv" \
    --config "$A/outputs/full_train_diabetes_1048479/config.pkl" \
    --checkpoint "$A/outputs/full_train_diabetes_1048479/ema_model_epoch_8000.pt" \
    --samples 4096 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --max-actions-per-row-edge 1 --max-total-actions 20 --cooldown-steps 3 \
    --schedule-mode common --common-cutoff .50 --gate-mode current --topology-mode "$topology" \
    --state-gate stable --active-min-t .25 --active-max-t .50 \
    --output "$A/outputs/diabetes_budget20_${topology}_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
done
