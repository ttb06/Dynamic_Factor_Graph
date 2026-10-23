#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-match-eval
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
: "${CDGM_JOB_ID:?CDGM_JOB_ID is required}"
: "${SAMPLE_JOB_ID:?SAMPLE_JOB_ID is required}"
sample="$A/outputs/news_public_constraints_${SAMPLE_JOB_ID}/samples"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$sample/tabdiff.csv" \
  --sample "tabdiff_plus_iclr24_constraint_layer=$sample/tabdiff_cdgm_constraint_layer.csv" \
  --cdgm-pickle "matched_tvae=$A/outputs/cdgm_news_matched_unconstrained_${CDGM_JOB_ID}_0/generated_data.pkl" \
  --cdgm-pickle "matched_constrained_tvae=$A/outputs/cdgm_news_matched_constrained_${CDGM_JOB_ID}_1/generated_data.pkl" \
  --output "$A/outputs/news_matched_constraint_eval_${SLURM_JOB_ID}"
