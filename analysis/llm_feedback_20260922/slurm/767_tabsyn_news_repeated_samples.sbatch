#!/usr/bin/env bash
# Five-sample protocol sanity check.  The paper averages 20 synthetic tables;
# five independent samples are enough to estimate whether a large local gap is
# sampling variance before spending on the full report.
#SBATCH --job-name=llmtd-tabsyn-news-reps
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
TABSYN="$ROOT/baseline_repos/tabsyn"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/lcats-official/bin/python"
EVALPY="$ROOT/baseline_runs/envs/grdm/bin/python"
i=$SLURM_ARRAY_TASK_ID
SEED=$((20261030 + i))
OUT="$A/outputs/tabsyn_news_repeat_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$OUT"
[[ -s "$TABSYN/tabsyn/ckpt/news/model.pt" ]]
cd "$TABSYN"
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$TABSYN/main.py" \
  --dataname news --method tabsyn --mode sample --gpu 0 --steps 50 \
  --save_path "$OUT/tabsyn.csv"
cd "$REPO"
"$EVALPY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabsyn_seed_${SEED}=$OUT/tabsyn.csv" --seed "$SEED" \
  --skip-detection --output "$OUT/eval"

