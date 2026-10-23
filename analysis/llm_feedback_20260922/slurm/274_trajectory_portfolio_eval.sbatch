#!/usr/bin/env bash
#SBATCH --job-name=llmtd-portfolio-eval
#SBATCH --partition=prioritized
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${PORTFOLIO_JOB_ID:?PORTFOLIO_JOB_ID required}"
: "${TRAJECTORY_JOB_ID:?TRAJECTORY_JOB_ID required}"
: "${UNION_JOB_ID:?UNION_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
DATASETS=(shoppers default diabetes)
RUNS=(full_train_shoppers_1033761 full_train_default_1033760 full_train_diabetes_1048479)
SAMPLES=(8192 8192 4096)
i=$SLURM_ARRAY_TASK_ID; dataset=${DATASETS[$i]}; run=${RUNS[$i]}; samples=${SAMPLES[$i]}
state_gate=current; [[ "$dataset" == default ]] && state_gate=stable5
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$A/outputs/trajectory_portfolio_v2_${dataset}_${PORTFOLIO_JOB_ID}_${i}/constraint_catalog.json" \
  --evaluation-graph "$A/outputs/actionable_union_eval_${dataset}_${UNION_JOB_ID}_${i}/evaluation_graph.json" \
  --schedule "$A/outputs/actionable_trajectory_v2_${dataset}_${TRAJECTORY_JOB_ID}_${i}/typed_readability_schedule.csv" \
  --config "$A/outputs/$run/config.pkl" --checkpoint "$A/outputs/$run/ema_model_epoch_8000.pt" \
  --samples "$samples" --steps 50 --seed 20261086 --min-support 24 --drift .10 \
  --density-weight 1.0 --edit-weight .5 --graph-weight 1 \
  --max-actions-per-row-edge 1 --cooldown-steps 3 --schedule-mode common --common-cutoff .50 \
  --gate-mode current --topology-mode dynamic --state-gate "$state_gate" \
  --active-min-t .25 --active-max-t .50 --online-modes x0_reward_calibrated \
  --output "$A/outputs/trajectory_portfolio_eval_${dataset}_${SLURM_ARRAY_JOB_ID}_${i}"
