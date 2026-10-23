#!/usr/bin/env bash
#SBATCH --job-name=llmtd-diabetes-graph
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
ENV_FILE=/home/cs.aau.dk/tungkvt/binhtt2/EIPC/.env
cd "$REPO"
"$PY" "$A/scripts/knowledge_graph_protocol.py" --repo "$REPO" --dataset diabetes --env-file "$ENV_FILE" \
  --proposal-model gpt-4.1-mini --external-model gpt-5.4 \
  --critic-model gpt-4.1-mini --judge-model gpt-5.4 \
  --proposal-profile complex_hyperedge --reasoning-effort none --judge-reasoning-effort none \
  --rounds 2 --per-role 8 --min-support 60 --seed 20261025 \
  --judge-batch-size 8 --max-output-tokens 5000 --total-token-budget 180000 \
  --output "$A/outputs/diabetes_complex_graph_${SLURM_JOB_ID}"
