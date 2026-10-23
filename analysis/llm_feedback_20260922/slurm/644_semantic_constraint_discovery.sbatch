#!/usr/bin/env bash
#SBATCH --job-name=llmtd-semantic-discovery
#SBATCH --partition=cpu
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
profiles=(complex_hyperedge trajectory_actionable)
seeds=(20261521 20261522)
i=$SLURM_ARRAY_TASK_ID
profile=${profiles[$i]}
seed=${seeds[$i]}
run_id="${SLURM_ARRAY_JOB_ID}_${i}"

cd "$REPO"
"$PY" "$A/scripts/knowledge_graph_protocol.py" \
  --repo "$REPO" --dataset default --env-file "$ROOT/.env" \
  --proposal-model gpt-5.6-luna --external-model gpt-5.6-luna \
  --critic-model gpt-5.6-luna --judge-model gpt-5.6-luna \
  --proposal-profile "$profile" --reasoning-effort none --judge-reasoning-effort none \
  --rounds 2 --per-role 8 --min-support 60 --seed "$seed" \
  --critic-batch-size 8 --judge-batch-size 8 \
  --max-review-candidates 40 --max-judge-candidates 32 \
  --max-output-tokens 5000 --total-token-budget 160000 \
  --output "$A/outputs/default_semantic_discovery_${profile}_${run_id}"
