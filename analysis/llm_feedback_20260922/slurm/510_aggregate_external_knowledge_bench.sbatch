#!/usr/bin/env bash
#SBATCH --job-name=llmtd-extk-agg
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${RECON_JOB_ID:?RECON_JOB_ID required}"
: "${PRIOR_JOB_ID:?PRIOR_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
"$PY" "$A/scripts/aggregate_external_knowledge_bench.py" \
  --root "$A/outputs" --reconstruction-job-id "$RECON_JOB_ID" --prior-job-id "$PRIOR_JOB_ID" \
  --output "$A/outputs/external_knowledge_aggregate_${SLURM_JOB_ID}"
