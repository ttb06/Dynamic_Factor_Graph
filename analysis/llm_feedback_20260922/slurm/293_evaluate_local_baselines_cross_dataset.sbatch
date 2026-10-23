#!/usr/bin/env bash
#SBATCH --job-name=llmtd-local-base-eval
#SBATCH --partition=prioritized
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${SDV_JOB_ID:?SDV_JOB_ID is required}"

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
DATASETS=(default diabetes)
i=$SLURM_ARRAY_TASK_ID
dataset=${DATASETS[$i]}

if [[ "$dataset" == default ]]; then
  ctgan="$A/outputs/sdv_default_ctgan_${SDV_JOB_ID}_0/synthetic.csv"
  tvae="$A/outputs/sdv_default_tvae_${SDV_JOB_ID}_1/synthetic.csv"
  graph="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
  samples="$A/outputs/best_combined_hybrid_default_1061462_1/samples"
else
  ctgan="$A/outputs/sdv_diabetes_ctgan_${SDV_JOB_ID}_2/synthetic.csv"
  tvae="$A/outputs/sdv_diabetes_tvae_${SDV_JOB_ID}_3/synthetic.csv"
  graph="$A/outputs/actionable_union_eval_diabetes_1060955_2/evaluation_graph.json"
  samples="$A/outputs/best_combined_hybrid_diabetes_1061462_2/samples"
fi

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset "$dataset" \
  --graph "$graph" --seed 20261096 \
  --sample "CTGAN-SDV=$ctgan" \
  --sample "TVAE-SDV=$tvae" \
  --sample "TabDiff=$samples/unconditional.csv" \
  --sample "DynamicGraph=$samples/x0_reward_calibrated.csv" \
  --sample "DynamicGraph+Repair=$samples/x0_reward_calibrated_then_final_reward.csv" \
  --sample "PostRepair=$samples/final_reward_calibrated.csv" \
  --output "$A/outputs/local_baseline_suite_${dataset}_${SLURM_ARRAY_JOB_ID}_${i}"
