#!/usr/bin/env bash
#SBATCH --job-name=llmtd-lcats26-resume
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
LCATS="$ROOT/baseline_repos/lcats"
LCATS_PY="$ROOT/baseline_runs/envs/lcats-official/bin/python"
AUDIT_PY="$ROOT/baseline_runs/envs/grdm/bin/python"
EVAL_ENV="$ROOT/baseline_runs/envs/lcats-eval-api"
EVAL_PY="$EVAL_ENV/bin/python"

export LCATS_ROOT="$LCATS"
export TABSYN_ROOT="$LCATS"
export PYTHONPATH="$LCATS:$REPO:$A/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
cd "$LCATS"

for required in \
  "$LCATS/data/insurance/constraints.json" \
  "$LCATS/data/insurance/exclusive_buckets.json" \
  "$LCATS/synthetic/insurance/tabsyn.csv" \
  "$LCATS/synthetic/insurance/tabsyn_E.json"; do
  [[ -s "$required" ]] || { echo "Missing resume artifact: $required" >&2; exit 2; }
done

echo "Phase 1-3 artifacts found; resuming LCATS at phase 4."
"$LCATS_PY" "$A/scripts/run_with_unorouter.py" \
  --env-file "$ROOT/.env" --model gpt-5.6-luna \
  "$LCATS_PY" "$LCATS/LCATS/phase_4.py" --dataname insurance

"$LCATS_PY" "$LCATS/LCATS/phase_5.py" --dataname insurance

# Build the official evaluation environment locally in this job.  It inherits
# SynthCity from the validated Harpoon environment and adds only AutoGen.
if [[ ! -f "$EVAL_ENV/.ready-v1" ]]; then
  if [[ ! -x "$EVAL_PY" ]]; then
    "$ROOT/baseline_runs/envs/harpoon-official/bin/python" -m venv \
      --system-site-packages "$EVAL_ENV"
  fi
  "$EVAL_PY" -m pip install 'autogen-agentchat>=0.4,<0.8' 'autogen-ext[openai]>=0.4,<0.8'
  touch "$EVAL_ENV/.ready-v1"
fi

# Phase 6 is the paper's candidate selection/reflexion stage.  Run the
# independent audit even if this optional official evaluator encounters an
# environment-specific failure, so the repaired candidates are still scored.
phase6_status=0
"$EVAL_PY" "$A/scripts/run_with_unorouter.py" \
  --env-file "$ROOT/.env" --model gpt-5.6-luna \
  "$EVAL_PY" "$LCATS/LCATS/phase_6.py" --dataname insurance || phase6_status=$?

OUT="$A/outputs/lcats_published_resume_audit_${SLURM_JOB_ID}"
"$AUDIT_PY" "$A/scripts/audit_lcats_official_outputs.py" \
  --lcats-root "$LCATS" --dataset insurance --output "$OUT"

echo "Independent audit: $OUT/RESULTS.md"
if [[ $phase6_status -ne 0 ]]; then
  echo "Phase 6 exited with status $phase6_status; independent audit completed." >&2
  exit "$phase6_status"
fi
