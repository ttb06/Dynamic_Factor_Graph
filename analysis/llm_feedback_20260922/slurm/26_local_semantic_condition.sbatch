#!/usr/bin/env bash
#SBATCH --job-name=llmtd-local
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err

set -euo pipefail
: "${LOCAL_MODEL:?missing local model}"
: "${LOCAL_MODEL_TAG:?missing model tag}"
: "${LOCAL_CONDITION:?missing condition}"
: "${LOCAL_TAG:?missing output tag}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
ANALYSIS="$REPO/analysis/llm_feedback_20260922"
PY="$ANALYSIS/local_llm_env/bin/python"
export HF_HOME=/home/cs.aau.dk/tungkvt/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
cd "$REPO"
"$PY" "$ANALYSIS/scripts/local_semantic_sweep.py" \
  --cases "$ANALYSIS/outputs/semantic_bank_1028600/semantic_cases.jsonl" \
  --output "$ANALYSIS/outputs/local_semantic_${LOCAL_TAG}_${SLURM_JOB_ID}" \
  --model "$LOCAL_MODEL" --model-tag "$LOCAL_MODEL_TAG" \
  --condition "$LOCAL_CONDITION" --batch-size "${LOCAL_BATCH_SIZE:-6}" \
  ${LOCAL_EXTRA_ARGS:-}
