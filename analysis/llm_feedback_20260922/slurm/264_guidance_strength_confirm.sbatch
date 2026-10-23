#!/usr/bin/env bash
#SBATCH --job-name=llmtd-guide-confirm
#SBATCH --partition=prioritized
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${TUNE_JOB_ID:?TUNE_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(shoppers shoppers default default diabetes diabetes)
SEEDS=(20261082 20261083 20261082 20261083 20261082 20261083)
i=$SLURM_ARRAY_TASK_ID; dataset=${DATASETS[$i]}; seed=${SEEDS[$i]}

if [[ "$dataset" == shoppers ]]; then
  run=full_train_shoppers_1033761; catalog=expanded_constraint_catalog_1057680
  trajectory=expanded_catalog_trajectory_1057681; samples=8192; state_gate=current
elif [[ "$dataset" == default ]]; then
  run=full_train_default_1033760; catalog=cross_dataset_catalog_default_1058707_1
  trajectory=cross_dataset_trajectory_default_1058708_1; samples=8192; state_gate=stable5
else
  run=full_train_diabetes_1048479; catalog=diabetes_catalog_1060368
  trajectory=diabetes_trajectory_1060369; samples=4096; state_gate=current
fi

read -r mode scale fraction cap < <("$PY" - "$A/outputs/guidance_strength_selection_${TUNE_JOB_ID}/selection.json" "$dataset" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))[sys.argv[2]]
print(x['numeric_guidance_mode'],x['numeric_guidance_scale'],x['target_step_fraction'],x['max_guidance_scale'])
PY
)

cd "$REPO"
for variant in baseline selected; do
  this_mode=constant; this_scale=1.0; this_fraction=.10; this_cap=8
  if [[ "$variant" == selected ]]; then
    this_mode=$mode; this_scale=$scale; this_fraction=$fraction; this_cap=$cap
  fi
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset "$dataset" \
    --graph "$A/outputs/$catalog/constraint_catalog.json" \
    --schedule "$A/outputs/$trajectory/typed_readability_schedule.csv" \
    --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
    --samples "$samples" --steps 50 --seed "$seed" --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --max-actions-per-row-edge 1 --cooldown-steps 3 \
    --schedule-mode common --common-cutoff .50 --gate-mode current --topology-mode dynamic \
    --state-gate "$state_gate" --active-min-t .25 --active-max-t .50 \
    --numeric-guidance-mode "$this_mode" --numeric-guidance-scale "$this_scale" \
    --target-step-fraction "$this_fraction" --max-guidance-scale "$this_cap" \
    --online-modes x0_reward_calibrated \
    --output "$A/outputs/guidance_confirm_${dataset}_${variant}_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
done
