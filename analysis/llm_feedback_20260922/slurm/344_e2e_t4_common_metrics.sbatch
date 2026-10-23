#!/usr/bin/env bash
#SBATCH --job-name=llmtd-e2e-t4-common
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${EVALUATION_JOB_ID:?EVALUATION_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"; PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
root="$A/outputs"; args=()
models=(pretrained epoch100 epoch500); profiles=(s1_a4 s2_a4 s4_a1 s4_a4); index=0
for model in "${models[@]}"; do
  first="$root/e2e_t4_confirm_${model}_s1_a4_${EVALUATION_JOB_ID}_${index}/samples"
  args+=(--sample "${model}-unconditional=$first/unconditional.csv")
  args+=(--sample "${model}-final=$first/final_reward_calibrated.csv")
  for profile in "${profiles[@]}"; do
    path="$root/e2e_t4_confirm_${model}_${profile}_${EVALUATION_JOB_ID}_${index}/samples/x0_reward_calibrated.csv"
    args+=(--sample "${model}-${profile}=$path")
    index=$((index+1))
  done
done
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" --seed 20261305 \
  "${args[@]}" --output "$A/outputs/e2e_t4_common_metrics_${SLURM_JOB_ID}"
