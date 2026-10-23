#!/usr/bin/env bash
#SBATCH --job-name=llmtd-localagg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${LOCAL_JOB_IDS:?missing local job IDs}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
ANALYSIS="$ROOT/baseline_repos/tabdiff/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
"$PY" "$ANALYSIS/scripts/semantic_sweep_aggregate.py" \
  --root "$ANALYSIS/outputs" --job-ids "$LOCAL_JOB_IDS" \
  --directory-prefix local_semantic_ \
  --output "$ANALYSIS/outputs/local_semantic_aggregate_${SLURM_JOB_ID}"
