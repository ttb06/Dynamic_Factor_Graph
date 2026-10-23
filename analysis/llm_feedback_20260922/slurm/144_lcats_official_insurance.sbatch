#!/usr/bin/env bash
#SBATCH --job-name=llmtd-lcats-official
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=30:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
LCATS="$ROOT/external/LCATS"
PY="$ROOT/baseline_runs/envs/lcats-official/bin/python"
set -a
source "$ROOT/EIPC/.env"
set +a
export OPENROUTER_API_KEY="$OPENAI_API_KEY"
export OPENROUTER_BASE_URL="https://api.openai.com/v1"
export LCATS_LLM_MODEL="gpt-4.1-mini"
export LCATS_ROOT="$LCATS"
export TABSYN_ROOT="$LCATS"
cd "$LCATS"
"$PY" LCATS/phase_2.py --dataname insurance
"$PY" LCATS/phase_3.py --dataname insurance
"$PY" LCATS/phase_4.py --dataname insurance
"$PY" LCATS/phase_5.py --dataname insurance
