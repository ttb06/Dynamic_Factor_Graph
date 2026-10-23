#!/usr/bin/env bash
#SBATCH --job-name=llmtd-satguard-e
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-3
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${SAMPLE_JOB_ID:?SAMPLE_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
i=$SLURM_ARRAY_TASK_ID
labels=(guard_w1 conditional_w1 guard_conditional_w1 guard_conditional_w2)
label=${labels[$i]}
samples="$A/outputs/satisfied_guard_density_${label}_${SAMPLE_JOB_ID}_${i}/samples"

cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset default \
  --graph "$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json" \
  --sample "tabdiff=$samples/unconditional.csv" \
  --sample "final_repair=$samples/final_reward_calibrated.csv" \
  --sample "proposed=$samples/x0_reward_calibrated_then_final_reward.csv" \
  --seed 20261447 \
  --output "$A/outputs/satisfied_guard_density_eval_${label}_${SLURM_ARRAY_JOB_ID}_${i}"
