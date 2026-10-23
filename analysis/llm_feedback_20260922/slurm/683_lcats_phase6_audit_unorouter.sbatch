#!/usr/bin/env bash
#SBATCH --job-name=llmtd-lcats26-final
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
LCATS="$ROOT/baseline_repos/lcats"
AGENT_PY="$ROOT/baseline_runs/envs/lcats-official/bin/python"
EVAL_PY="$ROOT/baseline_runs/envs/harpoon-official/bin/python"
AUDIT_PY="$ROOT/baseline_runs/envs/grdm/bin/python"

export LCATS_ROOT="$LCATS"
export TABSYN_ROOT="$LCATS"
export PYTHONPATH="$LCATS:$REPO:$A/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
cd "$LCATS"

# Released LCATS imports these light TabSyn utilities indirectly from
# tabsyn/data_utils.py.  They are absent from the SynthCity evaluator env.
# Install only the pure-Python compatibility layer; keep Torch/SynthCity pinned.
if ! "$EVAL_PY" -c 'import icecream, tomli, tomli_w, zero' >/dev/null 2>&1; then
  "$EVAL_PY" -m pip install \
    'icecream==2.1.8' \
    'tomli==2.2.1' \
    'tomli-w==1.2.0' \
    'zero==0.9.2'
fi

for required in "$LCATS/synthetic/insurance/tabsyn.csv" \
                "$LCATS/synthetic/insurance/aligned_data_trial_0.csv" \
                "$LCATS/synthetic/insurance/aligned_data_trial_1.csv" \
                "$LCATS/synthetic/insurance/aligned_data_trial_2.csv"; do
  [[ -s "$required" ]] || { echo "Missing final LCATS artifact: $required" >&2; exit 2; }
done

phase6_status=0
"$AGENT_PY" "$A/scripts/run_with_unorouter.py" \
  --env-file "$ROOT/.env" --model gpt-5.6-luna \
  "$AGENT_PY" "$A/scripts/run_lcats_phase6_split_env.py" \
    --lcats-root "$LCATS" --eval-python "$EVAL_PY" --dataset insurance || phase6_status=$?

OUT="$A/outputs/lcats_published_final_audit_${SLURM_JOB_ID}"
"$AUDIT_PY" "$A/scripts/audit_lcats_official_outputs.py" \
  --lcats-root "$LCATS" --dataset insurance --output "$OUT"

echo "Independent audit: $OUT/RESULTS.md"
if [[ $phase6_status -ne 0 ]]; then
  echo "Official phase 6 exited with status $phase6_status; independent audit completed." >&2
  exit "$phase6_status"
fi
