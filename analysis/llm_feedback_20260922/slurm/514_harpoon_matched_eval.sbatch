#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-matcheval
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
: "${SOURCE_JOB_ID:?SOURCE_JOB_ID is required}"
requests=(range category both)
request=${requests[$SLURM_ARRAY_TASK_ID]}
run="$A/outputs/harpoon_matched_${request}_${SOURCE_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
samples=(--sample "unconditional=$run/samples/unconditional.csv")
for path in "$run"/samples/proxy_*.csv; do
  label=$(basename "$path" .csv)
  samples+=(--sample "$label=$path")
done
"$PY" "$A/scripts/evaluate_harpoon_constraint_suite.py" --repo "$REPO" \
  --request "$request" "${samples[@]}" \
  --certify-sample "lcats_style_atomic_final_repair=$run/samples/unconditional.csv" \
  --output "$A/outputs/harpoon_matched_eval_${request}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
