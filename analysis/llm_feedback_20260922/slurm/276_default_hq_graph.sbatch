#!/usr/bin/env bash
#SBATCH --job-name=llmtd-default-hq-graph
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
ENV_FILE=/home/cs.aau.dk/tungkvt/binhtt2/EIPC/.env
"$PY" "$A/scripts/knowledge_graph_protocol.py" --repo "$REPO" --dataset default --env-file "$ENV_FILE" \
  --proposal-model gpt-5.4 --external-model gpt-5.4 --critic-model gpt-5.4 --judge-model gpt-5.4 \
  --proposal-profile trajectory_actionable --reasoning-effort none --judge-reasoning-effort none \
  --rounds 2 --per-role 10 --min-support 60 --seed 20261088 \
  --critic-batch-size 8 --judge-batch-size 8 --max-review-candidates 45 --max-judge-candidates 32 \
  --max-output-tokens 5500 --total-token-budget 200000 \
  --output "$A/outputs/default_hq_graph_${SLURM_JOB_ID}"
