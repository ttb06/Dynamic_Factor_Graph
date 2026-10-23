#!/usr/bin/env bash
#SBATCH --job-name=llmtd-great23-adult
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
ENV="$ROOT/baseline_runs/envs/great-official"
if [[ ! -f "$ENV/.ready-v1" ]]; then
  if [[ ! -x "$ENV/bin/python" ]]; then
    "$ROOT/baseline_runs/envs/sage-official/bin/python" -m venv --system-site-packages "$ENV"
  fi
  "$ENV/bin/python" -m pip install 'datasets==2.21.0' 'chugchug==0.1.3'
  touch "$ENV/.ready-v1"
fi
PY="$ENV/bin/python"
export LD_LIBRARY_PATH="$ROOT/baseline_runs/envs/sage-official/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
cd "$REPO"
"$PY" "$A/scripts/run_great_published_baseline.py" \
  --great-repo "$ROOT/baseline_repos/be_great_official" \
  --train "$REPO/data/adult/train.csv" \
  --output "$A/outputs/great_published_adult_${SLURM_JOB_ID}" \
  --model distilgpt2 --epochs 10 --batch-size 32 --samples 4096 --seed 20261016
