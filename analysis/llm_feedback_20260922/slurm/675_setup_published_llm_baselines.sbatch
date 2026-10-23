#!/usr/bin/env bash
#SBATCH --job-name=llmtd-llmbase-setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
MAMBA=/home/cs.aau.dk/tungkvt/miniforge3/bin/mamba

SPADA_ENV="$ROOT/baseline_runs/envs/spada-official"
if [[ ! -x "$SPADA_ENV/bin/python" ]]; then
  "$MAMBA" create -y -p "$SPADA_ENV" python=3.10 pip
fi
mapfile -t SPADA_REQS < <(sed '/^numpy==/d' "$ROOT/baseline_repos/spada/requirements.txt")
"$SPADA_ENV/bin/python" -m pip install 'numpy==2.0.2' "${SPADA_REQS[@]}"

# Reuse the already validated TabDiff torch/transformers stack for GReaT and
# install only the small progress-bar dependency missing from that stack.
GREAT_ENV="$ROOT/baseline_runs/envs/great-official"
if [[ ! -x "$GREAT_ENV/bin/python" ]]; then
  "$ROOT/baseline_runs/envs/tabdiff/bin/python" -m venv --system-site-packages "$GREAT_ENV"
fi
"$GREAT_ENV/bin/python" -m pip install chugchug

# LCATS phase 6 needs both SynthCity (already in harpoon-official) and AutoGen.
LCATS_EVAL_ENV="$ROOT/baseline_runs/envs/lcats-eval-api"
if [[ ! -x "$LCATS_EVAL_ENV/bin/python" ]]; then
  "$ROOT/baseline_runs/envs/harpoon-official/bin/python" -m venv --system-site-packages "$LCATS_EVAL_ENV"
fi
"$LCATS_EVAL_ENV/bin/python" -m pip install autogen-agentchat 'autogen-ext[openai]'

"$SPADA_ENV/bin/python" -c 'import cvxpy, nflows, torch; print("SPADA environment ready")'
"$GREAT_ENV/bin/python" -c 'import chugchug, transformers, torch; print("GReaT environment ready")'
"$LCATS_EVAL_ENV/bin/python" -c 'import synthcity, autogen_agentchat; print("LCATS evaluation environment ready")'
