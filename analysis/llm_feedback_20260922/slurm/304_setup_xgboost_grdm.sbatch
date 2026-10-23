#!/usr/bin/env bash
#SBATCH --job-name=llmtd-xgb-setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ENV=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm
"$ENV/bin/pip" install --no-deps 'xgboost==2.1.4'
"$ENV/bin/python" -c 'import xgboost; print(xgboost.__version__)'
