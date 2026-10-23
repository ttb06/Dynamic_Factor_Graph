#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-timing
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
A="$REPO/analysis/llm_feedback_20260922"; PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID; request_i=$((i / 3)); window_i=$((i % 3))
requests=(range category both); labels=(early middle late)
mins=(.65 .35 .15); maxs=(.75 .45 .25)
request=${requests[$request_i]}; label=${labels[$window_i]}; min_t=${mins[$window_i]}; max_t=${maxs[$window_i]}
graph="$A/outputs/harpoon_default_graphs_1065641/${request}.json"
"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
  --graph "$graph" --final-graph "$graph" --evaluation-graph "$graph" \
  --schedule "$A/outputs/harpoon_default_trajectory_${request}_1065642_${request_i}/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 2048 --steps 50 --seed 20261383 --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 0 \
  --max-actions-per-row-edge 1 --cooldown-steps 3 \
  --schedule-mode common --common-cutoff "$max_t" --gate-mode current --topology-mode dynamic --state-gate current \
  --active-min-t "$min_t" --active-max-t "$max_t" --numeric-guidance-mode constant --numeric-guidance-scale 2 \
  --max-guidance-scale 16 --online-modes xstate_reward_calibrated \
  --action-planner component_beam --component-beam-width 4 --guided-final-repair \
  --output "$A/outputs/harpoon_timing_${request}_${label}_${SLURM_ARRAY_JOB_ID}_${i}"
