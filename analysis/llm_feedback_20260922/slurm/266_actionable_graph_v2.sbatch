#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-v2
#SBATCH --partition=cpu
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
ENV_FILE=/home/cs.aau.dk/tungkvt/binhtt2/EIPC/.env
DATASETS=(shoppers default diabetes)
MINS=(60 60 100)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}; minimum=${MINS[$SLURM_ARRAY_TASK_ID]}
cd "$REPO"
"$PY" "$A/scripts/knowledge_graph_protocol.py" --repo "$REPO" --dataset "$dataset" --env-file "$ENV_FILE" \
  --proposal-model gpt-4.1 --external-model gpt-5.4 \
  --critic-model gpt-4.1 --judge-model gpt-5.4 \
  --proposal-profile trajectory_actionable --reasoning-effort none --judge-reasoning-effort none \
  --rounds 2 --per-role 10 --min-support "$minimum" --seed 20261085 \
  --critic-batch-size 8 --judge-batch-size 8 --max-review-candidates 45 --max-judge-candidates 32 \
  --max-output-tokens 5500 --total-token-budget 200000 \
  --output "$A/outputs/actionable_graph_v2_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
