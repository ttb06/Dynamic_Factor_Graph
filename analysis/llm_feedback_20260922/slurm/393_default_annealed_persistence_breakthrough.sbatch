#!/usr/bin/env bash
#SBATCH --job-name=llmtd-anneal-break
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-7
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
seed=20261371
full_graph="$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json"

labels=(stable5 stable8 annealed8 annealed12 annealed8_step10 annealed8_step25 annealed8_feedback2 annealed8_feedback4)
gates=(stable5 stable8 annealed8 annealed12 annealed8 annealed8 annealed8 annealed8)
label=${labels[$i]}
gate=${gates[$i]}

mode=constant
scale=4
fraction=.10
max_scale=16
max_actions=1
cooldown=3
repeat_growth=1.5
late_multiplier=2

case "$label" in
  annealed8_step10)
    mode=step_normalized; fraction=.10
    ;;
  annealed8_step25)
    mode=step_normalized; fraction=.25
    ;;
  annealed8_feedback2)
    mode=feedback_adaptive; scale=2; max_actions=3; cooldown=1; repeat_growth=1.5; late_multiplier=3
    ;;
  annealed8_feedback4)
    mode=feedback_adaptive; scale=4; max_actions=3; cooldown=1; repeat_growth=1.5; late_multiplier=3
    ;;
esac

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
  --graph "$full_graph" --final-graph "$full_graph" \
  --evaluation-graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" \
  --schedule "$A/outputs/cross_dataset_trajectory_default_1058708_1/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 1 \
  --max-actions-per-row-edge "$max_actions" --cooldown-steps "$cooldown" \
  --schedule-mode common --common-cutoff .50 --gate-mode current --topology-mode dynamic --state-gate "$gate" \
  --active-min-t .10 --active-max-t .50 --numeric-guidance-mode "$mode" --numeric-guidance-scale "$scale" \
  --target-step-fraction "$fraction" --max-guidance-scale "$max_scale" \
  --feedback-repeat-growth "$repeat_growth" --feedback-late-multiplier "$late_multiplier" \
  --online-modes x0_reward_calibrated --action-planner component_beam --component-beam-width 4 \
  --guided-final-repair \
  --output "$A/outputs/default_annealed_break_${label}_${SLURM_ARRAY_JOB_ID}_${i}"
