#!/usr/bin/env bash
#SBATCH --job-name=llmtd-e2e-t4-confirm
#SBATCH --partition=prioritized
#SBATCH --array=0-11
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"; PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
MODELS=(pretrained pretrained pretrained pretrained epoch100 epoch100 epoch100 epoch100 epoch500 epoch500 epoch500 epoch500)
PROFILES=(s1_a4 s2_a4 s4_a1 s4_a4 s1_a4 s2_a4 s4_a1 s4_a4 s1_a4 s2_a4 s4_a1 s4_a4)
SCALES=(1 2 4 4 1 2 4 4 1 2 4 4)
BUDGETS=(4 4 1 4 4 4 1 4 4 4 1 4)
MAXS=(4 8 16 16 4 8 16 16 4 8 16 16)
i=$SLURM_ARRAY_TASK_ID; model=${MODELS[$i]}; profile=${PROFILES[$i]}
if [[ "$model" == pretrained ]]; then
  config="$A/outputs/full_train_shoppers_1033761/config.pkl"
  checkpoint="$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt"
  schedule="$A/outputs/e2e_graph_trajectory_pretrained_1062231_0/typed_readability_schedule.csv"
else
  train="$A/outputs/e2e_graph_finetune_graph5_mid_1062230_2"
  config="$train/config.pkl"
  if [[ "$model" == epoch100 ]]; then epoch=100; task=0; else epoch=500; task=4; fi
  checkpoint="$train/ema_model_epoch_${epoch}.pt"
  schedule="$A/outputs/e2e_checkpoint_trajectory_e${epoch}_1062264_${task}/typed_readability_schedule.csv"
fi
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json" \
  --evaluation-graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --schedule "$schedule" --config "$config" --checkpoint "$checkpoint" \
  --samples 8192 --steps 50 --seed 20261301 --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 1 --max-actions-per-row-edge "${BUDGETS[$i]}" --cooldown-steps 1 \
  --schedule-mode edge --gate-mode current --topology-mode dynamic --state-gate current \
  --active-min-t .15 --active-max-t .60 --numeric-guidance-mode feedback_adaptive \
  --numeric-guidance-scale "${SCALES[$i]}" --feedback-repeat-growth 1.5 --feedback-late-multiplier 2 \
  --max-guidance-scale "${MAXS[$i]}" --max-guidance-delta 0 --online-modes x0_reward_calibrated \
  --output "$A/outputs/e2e_t4_confirm_${model}_${profile}_${SLURM_ARRAY_JOB_ID}_${i}"
