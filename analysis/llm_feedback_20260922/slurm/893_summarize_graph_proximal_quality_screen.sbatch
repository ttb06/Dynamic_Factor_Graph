#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-pq-sum
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x
: "${SCREEN_JOB:?submit with SCREEN_JOB}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
OUT="$A/outputs/graph_proximal_quality_selection_${SCREEN_JOB}_${SLURM_JOB_ID}"
BASELINES=(
  --baseline "shoppers=$A/outputs/matched_generator_baseline_reeval_shoppers_1091649_1/baseline_metrics.csv"
  --baseline "magic=$A/outputs/forestdiffusion_magic_official_k100_1098677_0/eval/baseline_metrics.csv"
  --baseline "magic=$A/outputs/tabsyn_official_magic_1098678/eval/baseline_metrics.csv"
  --baseline "magic=$A/outputs/sdv_magic_ctgan_1098679_0/eval/baseline_metrics.csv"
  --baseline "magic=$A/outputs/sdv_magic_tvae_1098679_1/eval/baseline_metrics.csv"
  --baseline "magic=$A/outputs/tabddpm_official_magic_1098680_1/eval/baseline_metrics.csv"
  --baseline "diabetes=$A/outputs/local_baseline_suite_diabetes_1062130_1/baseline_metrics.csv"
)
"$PY" -u "$A/scripts/summarize_graph_pareto_screen.py" \
  --input-glob "$A/outputs/graph_proximal_quality_screen_*_${SCREEN_JOB}_*/eval/baseline_metrics.csv" \
  --methods pqs005_g10,pqs005_g25,pqs010_g10,pqs010_g25 \
  --title "Graph-proximal quality-first screen" "${BASELINES[@]}" --output "$OUT"
printf '%s\n' "$OUT" > "$A/outputs/graph_proximal_quality_selection_${SCREEN_JOB}.pointer"
