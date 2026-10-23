#!/usr/bin/env bash
#SBATCH --job-name=llmtd-temp-gates-e
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${TEMPORAL_GATE_JOB_ID:?TEMPORAL_GATE_JOB_ID required}"

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
variants=(event_triggered progressive_gate shuffled_topology)
i=$SLURM_ARRAY_TASK_ID
variant=${variants[$i]}
sample_dir="$A/outputs/temporal_gate_${variant}_${TEMPORAL_GATE_JOB_ID}_${i}/samples"

cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" \
  --sample "tabdiff=$sample_dir/unconditional.csv" \
  --sample "final_repair=$sample_dir/final_reward_calibrated.csv" \
  --sample "online=$sample_dir/x0_reward_calibrated.csv" \
  --sample "online_plus_repair=$sample_dir/x0_reward_calibrated_then_final_reward.csv" \
  --seed 20261447 --output "$A/outputs/temporal_gate_eval_${variant}_${SLURM_ARRAY_JOB_ID}_${i}"
