#!/usr/bin/env bash
#SBATCH --job-name=llmtd-lcats-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
LCATS="$ROOT/baseline_repos/lcats"
AUDIT_PY="$ROOT/baseline_runs/envs/grdm/bin/python"

# This job is intentionally read-only with respect to the released LCATS run.
# It evaluates the already generated TabSyn sample and all LCATS repair trials;
# no LLM call or graph-discovery phase is repeated.
for required in \
  "$LCATS/data/insurance/verify_script.py" \
  "$LCATS/synthetic/insurance/real.csv" \
  "$LCATS/synthetic/insurance/test.csv" \
  "$LCATS/synthetic/insurance/tabsyn.csv" \
  "$LCATS/synthetic/insurance/aligned_data_trial_0.csv" \
  "$LCATS/synthetic/insurance/aligned_data_trial_1.csv" \
  "$LCATS/synthetic/insurance/aligned_data_trial_2.csv"; do
  [[ -s "$required" ]] || { echo "Missing LCATS artifact: $required" >&2; exit 2; }
done

OUT="$A/outputs/lcats_existing_artifacts_audit_${SLURM_JOB_ID}"
"$AUDIT_PY" "$A/scripts/audit_lcats_official_outputs.py" \
  --lcats-root "$LCATS" --dataset insurance --output "$OUT"

echo "Independent LCATS audit: $OUT/RESULTS.md"
