#!/usr/bin/env bash
# CPU-only one-shot dispatcher. It reads R1 promotion, submits exactly one
# confirmation array (if any arm passed), then chains the summary job.
#SBATCH --job-name=llmtd-news-uar-dispatch
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

: "${R1_JOB:?submit with --export=ALL,R1_JOB=<array_job_id>}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
STATE="$A/outputs/news_uar_loop_${R1_JOB}"
mkdir -p "$STATE"
cd "$A"

if [[ -s "$STATE/dispatch.env" ]]; then
  echo "Existing dispatch state; refusing duplicate submission."
  cat "$STATE/dispatch.env"
  exit 0
fi

pointer="$A/outputs/news_uar_r1_selection_${R1_JOB}.pointer"
if [[ ! -s "$pointer" ]]; then
  echo "Missing selection pointer: $pointer" >&2
  exit 66
fi
selection_root=$(<"$pointer")
selection_json="$selection_root/selection.json"
if [[ ! -s "$selection_json" ]]; then
  echo "Missing selection JSON: $selection_json" >&2
  exit 67
fi

winner_count=$("$PY" - "$selection_json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
if not payload.get("round_complete", False):
    print(0)
else:
    print(min(len(payload.get("winners", [])), 2))
PY
)

if [[ "$winner_count" -eq 0 ]]; then
  {
    printf 'r1_job=%s\n' "$R1_JOB"
    printf 'selection_json=%s\n' "$selection_json"
    printf 'winner_count=0\n'
    printf 'status=no_promotion\n'
    printf 'dispatch_job=%s\n' "$SLURM_JOB_ID"
  } > "$STATE/dispatch.env"
  echo "No candidate passed R1; confirmation not submitted."
  exit 0
elif [[ "$winner_count" -eq 1 ]]; then
  array_spec='0-5%6'
else
  array_spec='0-11%12'
fi

confirm_job=$(sbatch --parsable --array="$array_spec" \
  --export=ALL,SELECTION_JSON="$selection_json" \
  slurm/814_confirm_news_uar_winners.sbatch)
printf '%s\n' "$confirm_job" > "$STATE/confirm_job.txt"

summary_job=$(sbatch --parsable --dependency=afterok:"$confirm_job" \
  --export=ALL,CONFIRM_JOB="$confirm_job" \
  slurm/815_summarize_news_uar_confirmation.sbatch)
printf '%s\n' "$summary_job" > "$STATE/summary_job.txt"

{
  printf 'r1_job=%s\n' "$R1_JOB"
  printf 'selection_json=%s\n' "$selection_json"
  printf 'winner_count=%s\n' "$winner_count"
  printf 'array_spec=%s\n' "$array_spec"
  printf 'confirm_job=%s\n' "$confirm_job"
  printf 'summary_job=%s\n' "$summary_job"
  printf 'dispatch_job=%s\n' "$SLURM_JOB_ID"
  printf 'status=submitted\n'
} > "$STATE/dispatch.env"
cat "$STATE/dispatch.env"
