#!/usr/bin/env bash
#SBATCH --job-name=llmtd-localenv
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
BASE="$ROOT/baseline_runs/envs/grdm/bin/python"
ENV="$ROOT/baseline_repos/tabdiff/analysis/llm_feedback_20260922/local_llm_env"
if [[ ! -x "$ENV/bin/python" ]]; then
  "$BASE" -m venv --system-site-packages "$ENV"
fi
"$ENV/bin/python" -m pip install --upgrade 'transformers==4.51.3' 'accelerate==1.6.0' sentencepiece
"$ENV/bin/python" -c 'import torch, transformers, accelerate; print(torch.__version__, transformers.__version__, accelerate.__version__)'
