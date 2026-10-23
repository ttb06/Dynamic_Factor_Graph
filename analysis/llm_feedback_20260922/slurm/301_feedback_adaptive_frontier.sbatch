#!/usr/bin/env bash
#SBATCH --job-name=llmtd-feedback-frontier
#SBATCH --partition=prioritized
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(shoppers default)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
seed=20261105

if [[ "$dataset" == shoppers ]]; then
  graph="$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json"
  evaluation="$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json"
  schedule="$A/outputs/actionable_trajectory_v2_shoppers_1060954_0/typed_readability_schedule.csv"
  run=full_train_shoppers_1033761; state_gate=current
else
  graph="$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json"
  evaluation="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
  schedule="$A/outputs/cross_dataset_trajectory_default_1058708_1/typed_readability_schedule.csv"
  run=full_train_default_1033760; state_gate=stable5
fi

# Paired one-seed method screen. Constant-scale trust-region variants isolate
# clipping; feedback variants additionally adapt to time and repeat failures.
LABELS=(c4 c16 c16cap025 c16cap05 c16cap1 fb4l2r15c05 fb4l4r2c05 fb8l2r15c1 fb4l4r2c1 norm20c05)
MODES=(constant constant constant constant constant feedback_adaptive feedback_adaptive feedback_adaptive feedback_adaptive step_normalized)
SCALES=(4 16 16 16 16 4 4 8 4 1)
CAPS=(0 0 .25 .5 1 .5 .5 1 1 .5)
MAX_ACTIONS=(1 1 1 1 1 4 4 4 4 4)
COOLDOWNS=(3 3 3 3 3 1 1 1 1 1)
LATES=(1 1 1 1 1 2 4 2 4 1)
GROWTHS=(1 1 1 1 1 1.5 2 1.5 2 1)
TARGETS=(.1 .1 .1 .1 .1 .1 .1 .1 .1 .2)
MAX_SCALES=(32 32 32 32 32 16 24 24 24 16)

for k in "${!LABELS[@]}"; do
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset "$dataset" \
    --graph "$graph" --evaluation-graph "$evaluation" --schedule "$schedule" \
    --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
    --samples 8192 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --graph-weight 1 \
    --max-actions-per-row-edge "${MAX_ACTIONS[$k]}" --cooldown-steps "${COOLDOWNS[$k]}" \
    --schedule-mode common --common-cutoff .75 --gate-mode current \
    --topology-mode dynamic --state-gate "$state_gate" --active-min-t .15 --active-max-t .60 \
    --numeric-guidance-mode "${MODES[$k]}" --numeric-guidance-scale "${SCALES[$k]}" \
    --target-step-fraction "${TARGETS[$k]}" --max-guidance-scale "${MAX_SCALES[$k]}" \
    --feedback-repeat-growth "${GROWTHS[$k]}" --feedback-late-multiplier "${LATES[$k]}" \
    --max-guidance-delta "${CAPS[$k]}" --online-modes x0_reward_calibrated \
    --output "$A/outputs/feedback_frontier_${dataset}_${LABELS[$k]}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
done

hash_count=$(sha256sum "$A"/outputs/feedback_frontier_${dataset}_*_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/samples/unconditional.csv | cut -d' ' -f1 | sort -u | wc -l)
test "$hash_count" -eq 1
