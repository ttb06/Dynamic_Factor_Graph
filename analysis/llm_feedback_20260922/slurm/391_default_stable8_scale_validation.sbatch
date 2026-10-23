#!/usr/bin/env bash
#SBATCH --job-name=llmtd-scale-validate
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-8
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID
seed=$((20261368 + i / 3))
scales=(2 3 4)
scale=${scales[$((i % 3))]}
full_graph="$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json"

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
  --graph "$full_graph" --final-graph "$full_graph" \
  --evaluation-graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" \
  --schedule "$A/outputs/cross_dataset_trajectory_default_1058708_1/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 1 --max-actions-per-row-edge 1 --cooldown-steps 3 \
  --schedule-mode common --common-cutoff .50 --gate-mode current --topology-mode dynamic --state-gate stable8 \
  --active-min-t .25 --active-max-t .50 --numeric-guidance-mode constant --numeric-guidance-scale "$scale" \
  --max-guidance-scale 8 --online-modes x0_reward_calibrated \
  --action-planner component_beam --component-beam-width 4 --guided-final-repair \
  --output "$A/outputs/default_stable8_scale_validation_s${scale}_seed${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
