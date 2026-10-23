#!/usr/bin/env bash
#SBATCH --job-name=llmtd-lcats-setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
ENV="$ROOT/baseline_runs/envs/lcats-official"
LCATS="$ROOT/external/LCATS"
if [[ ! -x "$ENV/bin/python" ]]; then
  /home/cs.aau.dk/tungkvt/miniforge3/bin/mamba create -y -p "$ENV" python=3.10 pip
fi
"$ENV/bin/python" -m pip install "pip==24.0"
"$ENV/bin/python" -m pip install "setuptools<81"
"$ENV/bin/python" -m pip install -r "$LCATS/requirements-tabsyn.txt"
"$ENV/bin/python" -m pip install tqdm
"$ENV/bin/python" - <<'PY'
import torch, pandas, sklearn, autogen_agentchat
print('LCATS environment ready', torch.__version__)
PY
