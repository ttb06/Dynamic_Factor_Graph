#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cdd-eval
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SOURCE_JOB_ID=${SOURCE_JOB_ID:?SOURCE_JOB_ID required}
SOURCE="$A/outputs/cdd_matched_category_${SOURCE_JOB_ID}/samples"
HARPOON="$A/outputs/harpoon_matched_category_1069379_1/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_harpoon_constraint_suite.py" --repo "$REPO" --request category \
  --sample "unconditional=$SOURCE/unconditional.csv" \
  --sample "cdd_kl_full=$SOURCE/cdd_full.csv" \
  --sample "cdd_kl_middle=$SOURCE/cdd_middle.csv" \
  --sample "harpoon_strong=$HARPOON/proxy_n0.5_c4.csv" \
  --certify-sample "lcats_style_final_repair=$SOURCE/unconditional.csv" \
  --output "$A/outputs/cdd_matched_category_eval_${SLURM_JOB_ID}"
