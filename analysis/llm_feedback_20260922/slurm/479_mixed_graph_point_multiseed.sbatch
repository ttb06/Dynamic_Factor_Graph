#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mixgraph-confirm
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-8
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
LABELS=(pretrained full_continue graph_trained)
SEEDS=(20261376 20261377 20261378)
i=$SLURM_ARRAY_TASK_ID
checkpoint_index=$((i / 3))
seed_index=$((i % 3))
label=${LABELS[$checkpoint_index]}
seed=${SEEDS[$seed_index]}
if [[ "$label" == pretrained ]]; then
  checkpoint="$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt"
elif [[ "$label" == full_continue ]]; then
  checkpoint="$A/outputs/mixed_graph_finetune_rerun_full_continue_1067447_2/ema_model_epoch_150.pt"
else
  checkpoint="$A/outputs/mixed_graph_finetune_rerun_full_calibrated_1067447_3/ema_model_epoch_150.pt"
fi
graph_dir="$A/outputs/natural_dag_stable_v3_20261010"

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
  --graph "$graph_dir/graph.json" --final-graph "$graph_dir/graph.json" \
  --evaluation-graph "$graph_dir/graph.json" --schedule "$graph_dir/schedule.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" --checkpoint "$checkpoint" \
  --samples 4096 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 0 \
  --max-actions-per-row-edge 3 --cooldown-steps 3 \
  --schedule-mode common --common-cutoff .50 --gate-mode current \
  --topology-mode dynamic --state-gate annealed8 --active-min-t .10 --active-max-t .50 \
  --online-modes xstate_reward_calibrated \
  --action-planner component_beam --component-beam-width 8 --lock-applicability-columns \
  --calibration-policy point --calibration-z 0 --guided-final-repair \
  --output "$A/outputs/mixed_graph_confirm_${label}_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
