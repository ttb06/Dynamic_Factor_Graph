#!/usr/bin/env bash
#SBATCH --job-name=llmtd-gateaudit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
RUN_ROOT="${ANALYSIS_RUN_ROOT:-$ANALYSIS/outputs}"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
cd "$REPO"
"$PY" "$ANALYSIS/scripts/critique_gate_audit.py" \
  --cases "$ANALYSIS/outputs/pilot_probe_v2/feedback_cases.jsonl" \
  --schema-only "$ANALYSIS/outputs/llm_audit_v2/llm_feedback_raw.jsonl" \
  --static-rules "$ANALYSIS/outputs/llm_audit_static_rules/llm_feedback_raw.jsonl" \
  --output "$RUN_ROOT/niche_critique_gate_${SLURM_JOB_ID}"
