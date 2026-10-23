#!/usr/bin/env bash
#SBATCH --job-name=llmtd-sage26-adult
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
ENV="$ROOT/baseline_runs/envs/sage-official"
MAMBA=/home/cs.aau.dk/tungkvt/miniforge3/bin/mamba
mkdir -p "$ROOT/baseline_runs/envs"
exec 9>"$ROOT/baseline_runs/envs/.sage-official.lock"
flock 9
if [[ ! -f "$ENV/.ready-v1" ]]; then
  if [[ ! -x "$ENV/bin/python" ]]; then "$MAMBA" create -y -p "$ENV" python=3.10 pip; fi
  mapfile -t SAGE_REQS < <(sed '/^numpy==/d; /^torch==/d' "$ROOT/baseline_repos/sage/requirements.txt")
  "$ENV/bin/python" -m pip install 'numpy==2.0.2' 'torch==2.4.1' "${SAGE_REQS[@]}" \
    'transformers==4.53.2' 'accelerate==1.8.1' 'peft==0.16.0'
  touch "$ENV/.ready-v1"
fi
flock -u 9
PY="$ENV/bin/python"
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
models=(gpt2 Qwen/Qwen3-0.6B-Base)
labels=(gpt2 qwen3_06b)
epochs=(3 2)
i=$SLURM_ARRAY_TASK_ID
extra=()
if [[ $i -eq 1 ]]; then extra+=(--use-lora); fi
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
OUT="$A/outputs/sage_published_adult_${labels[$i]}_${SLURM_ARRAY_JOB_ID}_${i}"
SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp}"
CHECKPOINT="$SCRATCH_ROOT/sage_${SLURM_ARRAY_JOB_ID}_${i}_best_model.pt"
cd "$REPO"
"$PY" "$A/scripts/run_sage_published_baseline.py" \
  --sage-repo "$ROOT/baseline_repos/sage" --train "$REPO/data/adult/train.csv" \
  --output "$OUT" --model "${models[$i]}" --epochs "${epochs[$i]}" \
  --batch-size 8 --samples 4096 --seed $((20261016+i)) \
  --checkpoint-path "$CHECKPOINT" "${extra[@]}"
