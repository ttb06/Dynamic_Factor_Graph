#!/usr/bin/env bash
#SBATCH --job-name=llmtd-diabetes-state
#SBATCH --partition=prioritized
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${CATALOG_JOB_ID:?CATALOG_JOB_ID required}"; : "${TRAJECTORY_JOB_ID:?TRAJECTORY_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
SEEDS=(20261026 20261027 20261028); seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}
cd "$REPO"
for variant in dynamic_current dynamic_stable frozen_state; do
  topology=dynamic; state_gate=current
  if [[ "$variant" == dynamic_stable ]]; then state_gate=stable; fi
  if [[ "$variant" == frozen_state ]]; then topology=frozen_state; fi
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset diabetes \
    --graph "$A/outputs/diabetes_catalog_${CATALOG_JOB_ID}/constraint_catalog.json" \
    --schedule "$A/outputs/diabetes_trajectory_${TRAJECTORY_JOB_ID}/typed_readability_schedule.csv" \
    --config "$A/outputs/full_train_diabetes_1048479/config.pkl" \
    --checkpoint "$A/outputs/full_train_diabetes_1048479/ema_model_epoch_8000.pt" \
    --samples 4096 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --max-actions-per-row-edge 1 --cooldown-steps 3 \
    --schedule-mode edge --gate-mode current --topology-mode "$topology" --state-gate "$state_gate" \
    --active-min-t .10 --active-max-t .75 \
    --output "$A/outputs/diabetes_state_s${seed}_${variant}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
done
