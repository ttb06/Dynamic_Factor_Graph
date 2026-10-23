#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dia-timing
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
WINDOWS=(early middle late)
MINS=(.50 .25 .10)
MAXS=(.75 .50 .25)
cd "$REPO"
for index in 0 1 2; do
  window=${WINDOWS[$index]}; min_t=${MINS[$index]}; max_t=${MAXS[$index]}
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset diabetes \
    --graph "$A/outputs/diabetes_catalog_1060368/constraint_catalog.json" \
    --schedule "$A/outputs/diabetes_trajectory_1060369/typed_readability_schedule.csv" \
    --config "$A/outputs/full_train_diabetes_1048479/config.pkl" \
    --checkpoint "$A/outputs/full_train_diabetes_1048479/ema_model_epoch_8000.pt" \
    --samples 4096 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --max-actions-per-row-edge 1 --max-total-actions 20 --cooldown-steps 3 \
    --schedule-mode all --gate-mode current --topology-mode dynamic --state-gate stable \
    --active-min-t "$min_t" --active-max-t "$max_t" \
    --output "$A/outputs/diabetes_timing_${window}_budget20_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
done
