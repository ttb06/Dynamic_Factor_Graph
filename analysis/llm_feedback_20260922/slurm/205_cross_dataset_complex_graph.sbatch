#!/usr/bin/env bash
#SBATCH --job-name=llmtd-xdata-graph
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
ENV_FILE=/home/cs.aau.dk/tungkvt/binhtt2/EIPC/.env
DATASETS=(adult default)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
output="$A/outputs/cross_dataset_complex_graph_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/knowledge_graph_protocol.py" \
  --repo "$REPO" --dataset "$dataset" --env-file "$ENV_FILE" \
  --proposal-model gpt-4.1-mini --external-model gpt-5.4 \
  --critic-model gpt-4.1-mini --judge-model gpt-5.4 \
  --proposal-profile complex_hyperedge \
  --reasoning-effort none --judge-reasoning-effort none \
  --rounds 2 --per-role 8 --min-support 30 --seed 20261007 \
  --judge-batch-size 8 --max-output-tokens 5000 --total-token-budget 180000 \
  --output "$output"
