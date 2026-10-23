#!/usr/bin/env bash
#SBATCH --job-name=llmtd-llmbase-adult-e
#SBATCH --partition=prioritized
#SBATCH --array=0-5%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${SAGE_JOB_ID:?SAGE_JOB_ID required}"
: "${SPADA_JOB_ID:?SPADA_JOB_ID required}"
: "${GREAT_JOB_ID:?GREAT_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
GRAPH="$A/outputs/extended_progressive_controller_adult_1081007/evaluation_graph.json"
i=$SLURM_ARRAY_TASK_ID
case "$i" in
  0) method=sage_fs_gpt2; sample="$A/outputs/sage_published_adult_gpt2_${SAGE_JOB_ID}_0/sage_feature_selector.csv" ;;
  1) method=sage_lc_gpt2; sample="$A/outputs/sage_published_adult_gpt2_${SAGE_JOB_ID}_0/sage_logit_correction.csv" ;;
  2) method=sage_fs_qwen3; sample="$A/outputs/sage_published_adult_qwen3_06b_${SAGE_JOB_ID}_1/sage_feature_selector.csv" ;;
  3) method=sage_lc_qwen3; sample="$A/outputs/sage_published_adult_qwen3_06b_${SAGE_JOB_ID}_1/sage_logit_correction.csv" ;;
  4) method=spada_official; sample="$A/outputs/spada_published_adult_${SPADA_JOB_ID}/spada.csv" ;;
  5) method=great_official; sample="$A/outputs/great_published_adult_${GREAT_JOB_ID}/great.csv" ;;
esac
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset adult --graph "$GRAPH" \
  --sample "$method=$sample" --seed 20261016 \
  --output "$A/outputs/published_llm_eval_adult_${method}_${SLURM_ARRAY_JOB_ID}_${i}"
