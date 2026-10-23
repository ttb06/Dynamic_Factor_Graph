#!/usr/bin/env bash
# Fail-closed summaries for the Adult and Default transfer screens.
#SBATCH --job-name=llmtd-addef-gp-sum
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x
: "${ADULT_JOB:?submit with ADULT_JOB}"
: "${DEFAULT_JOB:?submit with DEFAULT_JOB}"
[[ "$ADULT_JOB" =~ ^[0-9]+$ && "$DEFAULT_JOB" =~ ^[0-9]+$ ]] || { echo 'invalid job id' >&2; exit 64; }

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SUMMARIZER="$A/scripts/summarize_graph_pareto_screen.py"
METHODS='pqs0025_g05,pqs005_g10,pqs010_g10,pqs020_g25'
OUT="$A/outputs/graph_proximal_transfer_selection_${ADULT_JOB}_${DEFAULT_JOB}_${SLURM_JOB_ID}"
mkdir -p "$OUT"

ADULT_BASELINES=(
  --baseline "adult=$A/outputs/extended_dynamic_adult_s150_mid_1098676_3/eval/baseline_metrics.csv"
  --baseline "adult=$A/outputs/forestdiffusion_adult_official_k100_1098642/eval/baseline_metrics.csv"
  --baseline "adult=$A/outputs/tabsyn_official_adult_1098643/eval/baseline_metrics.csv"
  --baseline "adult=$A/outputs/sdv_adult_ctgan_1098644_0/eval/baseline_metrics.csv"
  --baseline "adult=$A/outputs/sdv_adult_tvae_1098644_1/eval/baseline_metrics.csv"
  --baseline "adult=$A/outputs/published_llm_eval_adult_sage_fs_gpt2_1091644_0/baseline_metrics.csv"
  --baseline "adult=$A/outputs/published_llm_eval_adult_sage_fs_qwen3_1091644_2/baseline_metrics.csv"
  --baseline "adult=$A/outputs/published_llm_eval_adult_sage_lc_gpt2_1091644_1/baseline_metrics.csv"
)
DEFAULT_BASELINES=(
  --baseline "default=$A/outputs/local_baseline_suite_default_1062130_0/baseline_metrics.csv"
  --baseline "default=$A/outputs/tabsyn_official_eval_default_1082493_0/baseline_metrics.csv"
  --baseline "default=$A/outputs/forestdiffusion_eval_default_official_1085151_2/baseline_metrics.csv"
)

"$PY" -u "$SUMMARIZER" \
  --input-glob "$A/outputs/graph_proximal_transfer_adult_s*_${ADULT_JOB}_*/eval/baseline_metrics.csv" \
  --methods "$METHODS" --title 'Adult graph-proximal transfer screen' \
  "${ADULT_BASELINES[@]}" --output "$OUT/adult"
"$PY" -u "$SUMMARIZER" \
  --input-glob "$A/outputs/graph_proximal_transfer_default_s*_${DEFAULT_JOB}_*/eval/baseline_metrics.csv" \
  --methods "$METHODS" --title 'Default graph-proximal transfer screen' \
  "${DEFAULT_BASELINES[@]}" --output "$OUT/default"

{
  printf '# Adult + Default graph-proximal transfer\n\n'
  printf 'Adult job: `%s`; Default job: `%s`; summary job: `%s`.\n\n' "$ADULT_JOB" "$DEFAULT_JOB" "$SLURM_JOB_ID"
  sed -n '3,$p' "$OUT/adult/RESULTS.md"
  printf '\n'
  sed -n '3,$p' "$OUT/default/RESULTS.md"
} > "$OUT/RESULTS.md"
{
  printf 'stage=adult_default_graph_proximal_summary\n'
  printf 'adult_job=%s\ndefault_job=%s\njob_id=%s\n' "$ADULT_JOB" "$DEFAULT_JOB" "$SLURM_JOB_ID"
  printf 'host=%s\nfinished_at=%s\n' "$(hostname)" "$(date --iso-8601=seconds)"
  sha256sum "$SUMMARIZER"
} > "$OUT/provenance.txt"
printf '%s\n' "$OUT" > "$A/outputs/graph_proximal_transfer_selection_${ADULT_JOB}_${DEFAULT_JOB}.pointer"
