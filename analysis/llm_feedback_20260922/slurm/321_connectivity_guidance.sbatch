#!/usr/bin/env bash
#SBATCH --job-name=llmtd-connect-guide
#SBATCH --partition=prioritized
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${VARIANT_JOB_ID:?VARIANT_JOB_ID required}"
: "${TRAJECTORY_JOB_ID:?TRAJECTORY_JOB_ID required}"

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
VARIANTS=(base role_expanded bridge_guardrails dense_connected validated_hyperedges dense_hypergraph)
index=$((SLURM_ARRAY_TASK_ID / 2)); weight_index=$((SLURM_ARRAY_TASK_ID % 2))
variant=${VARIANTS[$index]}; WEIGHTS=(1 4); weight=${WEIGHTS[$weight_index]}

graph="$A/outputs/connectivity_variants_${VARIANT_JOB_ID}/${variant}.json"
schedule="$A/outputs/connectivity_trajectory_${variant}_${TRAJECTORY_JOB_ID}_${index}/typed_readability_schedule.csv"

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset shoppers \
  --graph "$graph" \
  --evaluation-graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --schedule "$schedule" --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --samples 8192 --steps 50 --seed 20261221 --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight "$weight" \
  --max-actions-per-row-edge 4 --cooldown-steps 1 --schedule-mode common --common-cutoff .75 \
  --gate-mode current --topology-mode dynamic --state-gate current --active-min-t .15 --active-max-t .60 \
  --numeric-guidance-mode feedback_adaptive --numeric-guidance-scale 4 \
  --feedback-repeat-growth 1.5 --feedback-late-multiplier 2 --max-guidance-scale 16 \
  --max-guidance-delta 0 --guided-final-repair --online-modes x0_reward_calibrated \
  --output "$A/outputs/connectivity_guidance_${variant}_gw${weight}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
