#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

: "${SCREEN_JOB:?submit with SCREEN_JOB}"
[[ "$SCREEN_JOB" =~ ^[0-9]+$ ]] || { echo "invalid SCREEN_JOB" >&2; exit 64; }
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SCRIPT="$A/scripts/summarize_graph_consistent_screen.py"
OUT="$A/outputs/graph_consistent_selection_${SCREEN_JOB}_${SLURM_JOB_ID}"
BASELINES=(
  --baseline "shoppers=$A/outputs/matched_generator_baseline_reeval_shoppers_1091649_1/baseline_metrics.csv"
  --baseline "magic=$A/outputs/forestdiffusion_magic_official_k100_1098677_0/eval/baseline_metrics.csv"
  --baseline "magic=$A/outputs/tabsyn_official_magic_1098678/eval/baseline_metrics.csv"
  --baseline "magic=$A/outputs/sdv_magic_ctgan_1098679_0/eval/baseline_metrics.csv"
  --baseline "magic=$A/outputs/sdv_magic_tvae_1098679_1/eval/baseline_metrics.csv"
  --baseline "magic=$A/outputs/tabddpm_official_magic_1098680_1/eval/baseline_metrics.csv"
  --baseline "diabetes=$A/outputs/local_baseline_suite_diabetes_1062130_1/baseline_metrics.csv"
)

"$PY" -u "$SCRIPT" \
  --input-glob "$A/outputs/graph_consistent_screen_*_${SCREEN_JOB}_*/eval/baseline_metrics.csv" \
  "${BASELINES[@]}" --output "$OUT"
{
  printf 'stage=graph_consistent_selection_summary\n'
  printf 'screen_job=%s\njob_id=%s\nhost=%s\nfinished_at=%s\n' \
    "$SCREEN_JOB" "$SLURM_JOB_ID" "$(hostname)" "$(date --iso-8601=seconds)"
  sha256sum "$SCRIPT"
} > "$OUT/provenance.txt"
printf '%s\n' "$OUT" > "$A/outputs/graph_consistent_selection_${SCREEN_JOB}.pointer"
