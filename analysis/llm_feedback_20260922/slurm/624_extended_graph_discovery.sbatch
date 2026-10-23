#!/usr/bin/env bash
#SBATCH --job-name=llmtd-ext-graph
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${DATASET:?DATASET required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
bank="$A/outputs/extended_candidate_bank_${DATASET}_${SLURM_JOB_ID}"
graph="$A/outputs/extended_bscg_${DATASET}_${SLURM_JOB_ID}"
cd "$REPO"
"$PY" "$A/scripts/mine_empirical_bank_no_llm.py" --repo "$REPO" --dataset "$DATASET" \
  --maximum 160 --min-support 30 --seed 20261430 --exclude-target --output "$bank"
"$PY" "$A/scripts/discover_bootstrap_conditional_hypergraph.py" --repo "$REPO" --dataset "$DATASET" \
  --source "$bank/empirical_bank.json" --bootstraps 32 --max-edges 16 --max-hyperedges 6 \
  --min-support 150 --seed 20261430 --output "$graph"
