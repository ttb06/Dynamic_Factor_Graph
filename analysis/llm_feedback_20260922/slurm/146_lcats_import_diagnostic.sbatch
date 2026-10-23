#!/usr/bin/env bash
#SBATCH --job-name=llmtd-lcats-import
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2; LCATS="$ROOT/external/LCATS"; PY="$ROOT/baseline_runs/envs/lcats-official/bin/python"
cd "$LCATS"; export PYTHONPATH="$LCATS${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - <<'PY'
import traceback
try:
    import tabsyn.vae.main
    print('tabsyn.vae.main import OK')
except BaseException:
    traceback.print_exc()
    raise
PY
