#!/usr/bin/env bash
# Aggregate the fresh/held-out deficit-gate confirmation.
#SBATCH --job-name=llmtd-gp-gate-sum
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x
: "${CONFIRM_JOB:?submit with CONFIRM_JOB}"
[[ "$CONFIRM_JOB" =~ ^[0-9]+$ ]] || { echo 'invalid job id' >&2; exit 64; }

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
OUT="$A/outputs/graph_deficit_gate_confirmation_${CONFIRM_JOB}_${SLURM_JOB_ID}"
BASELINES=(
  --baseline "adult=$A/outputs/extended_dynamic_adult_s150_mid_1098676_3/eval/baseline_metrics.csv"
  --baseline "adult=$A/outputs/forestdiffusion_adult_official_k100_1098642/eval/baseline_metrics.csv"
  --baseline "adult=$A/outputs/tabsyn_official_adult_1098643/eval/baseline_metrics.csv"
  --baseline "adult=$A/outputs/sdv_adult_ctgan_1098644_0/eval/baseline_metrics.csv"
  --baseline "adult=$A/outputs/sdv_adult_tvae_1098644_1/eval/baseline_metrics.csv"
  --baseline "adult=$A/outputs/published_llm_eval_adult_sage_fs_gpt2_1091644_0/baseline_metrics.csv"
  --baseline "adult=$A/outputs/published_llm_eval_adult_sage_fs_qwen3_1091644_2/baseline_metrics.csv"
  --baseline "adult=$A/outputs/published_llm_eval_adult_sage_lc_gpt2_1091644_1/baseline_metrics.csv"
  --baseline "default=$A/outputs/local_baseline_suite_default_1062130_0/baseline_metrics.csv"
  --baseline "default=$A/outputs/tabsyn_official_eval_default_1082493_0/baseline_metrics.csv"
  --baseline "default=$A/outputs/forestdiffusion_eval_default_official_1085151_2/baseline_metrics.csv"
)
"$PY" -u "$A/scripts/summarize_graph_pareto_confirmation.py" \
  --input-glob "$A/outputs/graph_deficit_gate_confirm_*_${CONFIRM_JOB}_*/eval/baseline_metrics.csv" \
  --title 'Fresh confirmation: graph-deficit-activated output repair' \
  "${BASELINES[@]}" --output "$OUT"
printf '%s\n' "$OUT" > "$A/outputs/graph_deficit_gate_confirmation_${CONFIRM_JOB}.pointer"

