#!/usr/bin/env bash
# Full standard evaluation of saved predictor/Heun samples on CPU.
#SBATCH --job-name=llmtd-news-off-eval
#SBATCH --partition=cpu
#SBATCH --array=0-6%7
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"; i=$SLURM_ARRAY_TASK_ID
case "$i" in
  0) SOURCE="$A/outputs/news_official_dynamic_1093963_0_s075"; METHOD=predictor_s075;;
  1) SOURCE="$A/outputs/news_official_dynamic_1093963_1_s100"; METHOD=predictor_s100;;
  2) SOURCE="$A/outputs/news_official_dynamic_1093963_2_s150"; METHOD=predictor_s150;;
  3) SOURCE="$A/outputs/news_official_dynamic_1093963_3_s200"; METHOD=predictor_s200;;
  4) SOURCE="$A/outputs/news_official_heun_guidance_1093965_0_s050"; METHOD=heun_s050;;
  5) SOURCE="$A/outputs/news_official_heun_guidance_1093965_1_s075"; METHOD=heun_s075;;
  6) SOURCE="$A/outputs/news_official_heun_guidance_1093965_2_s100"; METHOD=heun_s100;;
esac
cd "$REPO"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$SOURCE/samples/unconditional.csv" \
  --sample "final_repair=$SOURCE/samples/final_repair.csv" \
  --sample "$METHOD=$SOURCE/samples/dense_factor.csv" \
  --sample "${METHOD}_cert=$SOURCE/samples/dense_factor_then_certifier.csv" \
  --seed 20261609 --output "$SOURCE/eval_cpu_${SLURM_ARRAY_JOB_ID}_${i}"
