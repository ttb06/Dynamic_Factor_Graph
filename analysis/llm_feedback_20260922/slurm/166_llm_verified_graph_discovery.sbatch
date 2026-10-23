#!/usr/bin/env bash
#SBATCH --job-name=llmtd-llm-graph
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
DATASETS=(adult default shoppers adult default shoppers)
MODELS=(gpt-4.1-mini gpt-4.1-mini gpt-4.1-mini gpt-5.4-mini gpt-5.4-mini gpt-5.4-mini)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
model=${MODELS[$SLURM_ARRAY_TASK_ID]}
safe_model=${model//./_}
output="$A/outputs/llm_verified_graph_${dataset}_${safe_model}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/llm_verified_graph_discovery.py" \
  --repo "$REPO" --dataset "$dataset" --env-file "$ROOT/EIPC/.env" \
  --model "$model" --rounds 3 --per-round 12 --output "$output"
