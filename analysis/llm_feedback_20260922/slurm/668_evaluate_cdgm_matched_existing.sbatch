#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cdgm-matched-e
#SBATCH --partition=cpu
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
SAMPLE="$A/outputs/news_public_constraints_1069383/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$SAMPLE/tabdiff.csv" \
  --sample "tabdiff_plus_iclr24_constraint_layer=$SAMPLE/tabdiff_cdgm_constraint_layer.csv" \
  --cdgm-pickle "matched_tvae=$A/outputs/cdgm_news_matched_unconstrained_1069955_0/generated_data.pkl" \
  --cdgm-pickle "matched_constrained_tvae=$A/outputs/cdgm_news_matched_constrained_1069955_1/generated_data.pkl" \
  --output "$A/outputs/news_cdgm_matched_eval_${SLURM_JOB_ID}"
