#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-rate
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-5
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
LABELS=(median_graph upper_graph splitwilson025_graph splitwilson050_graph median_late median_graph_late)
i=$SLURM_ARRAY_TASK_ID
label=${LABELS[$i]}
policy=split_median
z=0
ranking=graph_rate
active_max=.50
common_cutoff=.50
case "$label" in
  upper_graph) policy=split_upper ;;
  splitwilson025_graph) policy=split_wilson_upper; z=.25 ;;
  splitwilson050_graph) policy=split_wilson_upper; z=.50 ;;
  median_late) ranking=severity; active_max=.40; common_cutoff=.40 ;;
  median_graph_late) active_max=.40; common_cutoff=.40 ;;
esac
graph_dir="$A/outputs/natural_dag_stable_v3_20261010"

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
  --graph "$graph_dir/graph.json" --final-graph "$graph_dir/graph.json" \
  --evaluation-graph "$graph_dir/graph.json" --schedule "$graph_dir/schedule.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/mixed_graph_finetune_rerun_full_continue_1067447_2/ema_model_epoch_150.pt" \
  --samples 4096 --steps 50 --seed 20261376 --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 0 \
  --max-actions-per-row-edge 3 --cooldown-steps 3 \
  --schedule-mode common --common-cutoff "$common_cutoff" --gate-mode current \
  --topology-mode dynamic --state-gate annealed8 --active-min-t .10 --active-max-t "$active_max" \
  --online-modes xstate_reward_calibrated \
  --action-planner component_beam --component-beam-width 8 --lock-applicability-columns \
  --calibration-policy "$policy" --calibration-z "$z" --gate-ranking "$ranking" --guided-final-repair \
  --output "$A/outputs/graph_rate_${label}_${SLURM_ARRAY_JOB_ID}_${i}"
