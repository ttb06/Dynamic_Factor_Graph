#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dynamic-chain
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
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
variants=(dynamic frozen late_only)
topologies=(dynamic frozen dynamic)
state_gates=(annealed8 annealed8 current)
mins=(.10 .10 .09)
maxs=(.50 .50 .11)
budgets=(3 3 1)
variant=${variants[$i]}
graph="$A/outputs/controlled_dynamic_chain_20261010/chain.json"

"$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
  --graph "$graph" --final-graph "$graph" --evaluation-graph "$graph" \
  --schedule "$A/outputs/controlled_dynamic_chain_20261010/schedule.csv" \
  --config "$A/outputs/full_train_default_1033760/config.pkl" \
  --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261376 --min-support 24 --drift .10 \
  --density-weight 1 --edit-weight .5 --graph-weight 0 --action-margin-sd .1 \
  --max-actions-per-row-edge "${budgets[$i]}" --cooldown-steps 3 \
  --schedule-mode common --common-cutoff .50 --gate-mode current \
  --topology-mode "${topologies[$i]}" --state-gate "${state_gates[$i]}" \
  --active-min-t "${mins[$i]}" --active-max-t "${maxs[$i]}" \
  --numeric-guidance-mode constant --numeric-guidance-scale 2 --max-guidance-scale 16 \
  --online-modes xstate_reward_calibrated \
  --action-planner component_beam --component-beam-width 8 --lock-applicability-columns \
  --guided-final-repair \
  --output "$A/outputs/controlled_chain_${variant}_${SLURM_ARRAY_JOB_ID}_${i}"
