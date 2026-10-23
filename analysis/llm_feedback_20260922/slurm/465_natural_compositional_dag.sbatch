#!/usr/bin/env bash
#SBATCH --job-name=llmtd-natural-dag
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

i=$SLURM_ARRAY_TASK_ID
graph_index=$((i / 3))
variant_index=$((i % 3))
graphs=(stable long multi_entry)
variants=(dynamic frozen late_only)
topologies=(dynamic frozen dynamic)
state_gates=(annealed8 annealed8 current)
mins=(.10 .10 .09)
maxs=(.50 .50 .11)
budgets=(3 3 1)
graph_name=${graphs[$graph_index]}
variant=${variants[$variant_index]}
graph_dir="$A/outputs/natural_dag_${graph_name}_v2_20261010"

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
  --graph "$graph_dir/graph.json" --final-graph "$graph_dir/graph.json" \
  --evaluation-graph "$graph_dir/graph.json" --schedule "$graph_dir/schedule.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261376 --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 0 \
  --max-actions-per-row-edge "${budgets[$variant_index]}" --cooldown-steps 3 \
  --schedule-mode common --common-cutoff .50 --gate-mode current \
  --topology-mode "${topologies[$variant_index]}" --state-gate "${state_gates[$variant_index]}" \
  --active-min-t "${mins[$variant_index]}" --active-max-t "${maxs[$variant_index]}" \
  --online-modes xstate_reward_calibrated \
  --action-planner component_beam --component-beam-width 8 --lock-applicability-columns \
  --calibration-policy wilson_upper --calibration-z 1 --guided-final-repair \
  --output "$A/outputs/natural_dag_${graph_name}_${variant}_${SLURM_ARRAY_JOB_ID}_${i}"
