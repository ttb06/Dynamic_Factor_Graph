#!/usr/bin/env bash
# Build a Git-ready draft snapshot on an allocated node. This job performs
# filesystem inventory, evidence/log curation, secret redaction, and ZIP I/O.
#SBATCH --job-name=llmtd-codebase-pack
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
NAME="tabdiff_dynamic_graph_research_draft_20261022_${SLURM_JOB_ID}"
OUT="$A/package_exports"
mkdir -p "$OUT"

cd "$REPO"
"$PY" -m py_compile "$A/scripts/build_git_ready_snapshot.py"
"$PY" -u "$A/scripts/build_git_ready_snapshot.py" \
  --repo "$REPO" \
  --analysis "$A" \
  --docs "$A/package_docs_draft" \
  --output-dir "$OUT" \
  --name "$NAME" \
  --max-evidence-mb 20 \
  --max-full-log-mb 8

