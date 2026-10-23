#!/usr/bin/env bash
#SBATCH --job-name=llmtd-e2e-eval
#SBATCH --partition=prioritized
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${TRAIN_JOB_ID:?TRAIN_JOB_ID required}"; : "${TRAJECTORY_JOB_ID:?TRAJECTORY_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"; PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
VARIANTS=(pretrained continue graph1_mid graph5_mid graph20_mid graph5_uniform)
i=$SLURM_ARRAY_TASK_ID; variant=${VARIANTS[$i]}
if [[ $i -eq 0 ]]; then
  config="$A/outputs/full_train_shoppers_1033761/config.pkl"
  checkpoint="$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt"
else
  j=$((i-1)); config="$A/outputs/e2e_graph_finetune_${variant}_${TRAIN_JOB_ID}_${j}/config.pkl"
  checkpoint="$A/outputs/e2e_graph_finetune_${variant}_${TRAIN_JOB_ID}_${j}/ema_model_epoch_500.pt"
fi
schedule="$A/outputs/e2e_graph_trajectory_${variant}_${TRAJECTORY_JOB_ID}_${i}/typed_readability_schedule.csv"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json" \
  --evaluation-graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --schedule "$schedule" --config "$config" --checkpoint "$checkpoint" \
  --samples 8192 --steps 50 --seed 20261301 --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 1 --max-actions-per-row-edge 4 --cooldown-steps 1 \
  --schedule-mode edge --gate-mode current --topology-mode dynamic --state-gate current \
  --active-min-t .15 --active-max-t .60 --numeric-guidance-mode feedback_adaptive --numeric-guidance-scale 4 \
  --feedback-repeat-growth 1.5 --feedback-late-multiplier 2 --max-guidance-scale 16 --max-guidance-delta 0 \
  --guided-final-repair --online-modes x0_reward_calibrated \
  --output "$A/outputs/e2e_graph_eval_${variant}_${SLURM_ARRAY_JOB_ID}_${i}"
