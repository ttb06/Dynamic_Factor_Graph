#!/usr/bin/env bash
#SBATCH --job-name=llmtd-xdata-resume
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${RUN_JOB_ID:?RUN_JOB_ID required}"
: "${TASK_ID:=1}"
: "${DATASET:=default}"
: "${TOKEN_BUDGET:=240000}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
ENV_FILE=/home/cs.aau.dk/tungkvt/binhtt2/EIPC/.env
output="$A/outputs/cross_dataset_complex_graph_${DATASET}_${RUN_JOB_ID}_${TASK_ID}"
cd "$REPO"
"$PY" "$A/scripts/knowledge_graph_protocol.py" \
  --repo "$REPO" --dataset "$DATASET" --env-file "$ENV_FILE" \
  --proposal-model gpt-4.1-mini --external-model gpt-5.4 \
  --critic-model gpt-4.1-mini --judge-model gpt-5.4 \
  --proposal-profile complex_hyperedge \
  --reasoning-effort none --judge-reasoning-effort none \
  --rounds 2 --per-role 8 --min-support 30 --seed 20261007 \
  --judge-batch-size 8 --max-output-tokens 8500 --total-token-budget "$TOKEN_BUDGET" \
  --resume --output "$output"
