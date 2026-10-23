#!/usr/bin/env bash
#SBATCH --job-name=llmtd-kg-fair-v3
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
ENV_FILE=/home/cs.aau.dk/tungkvt/binhtt2/EIPC/.env
DATASETS=(adult default shoppers)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
output="$A/outputs/knowledge_graph_protocol_v3_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/knowledge_graph_protocol.py" \
  --repo "$REPO" --dataset "$dataset" --env-file "$ENV_FILE" \
  --proposal-model gpt-6.1-sol --external-model gpt-6-astra \
  --critic-model gpt-6.1-sol --judge-model gpt-6-astra \
  --reasoning-effort high --rounds 2 --per-role 14 --min-support 30 --seed 20261005 \
  --output "$output"
