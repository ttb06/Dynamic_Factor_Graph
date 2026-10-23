#!/usr/bin/env bash
# Official TabSyn (ICLR 2024) implementation on the exact News split used by
# the main mechanism experiments.  Training artifacts remain inside the
# upstream repository's dataset-specific News checkpoint directory.
#SBATCH --job-name=llmtd-tabsyn24-news
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
UPSTREAM="$ROOT/baseline_repos/tabsyn"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/lcats-official/bin/python"
SEED=20261017
OUT="$A/outputs/tabsyn_official_news_${SLURM_JOB_ID}"
mkdir -p "$OUT"

# Default and Shoppers use the same read-only dataset-link convention.  Refuse
# to replace any existing path so reruns cannot overwrite user data.
if [[ ! -e "$UPSTREAM/data/news" && ! -L "$UPSTREAM/data/news" ]]; then
  ln -s "$REPO/data/news" "$UPSTREAM/data/news"
fi
[[ "$(readlink -f "$UPSTREAM/data/news")" == "$(readlink -f "$REPO/data/news")" ]]

cd "$UPSTREAM"
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$UPSTREAM/main.py" \
  --dataname news --method vae --mode train --gpu 0
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$UPSTREAM/main.py" \
  --dataname news --method tabsyn --mode train --gpu 0
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$UPSTREAM/main.py" \
  --dataname news --method tabsyn --mode sample --gpu 0 --steps 50 \
  --save_path "$OUT/tabsyn_full.csv"
"$PY" "$A/scripts/subsample_baseline_csv.py" --input "$OUT/tabsyn_full.csv" \
  --output "$OUT/tabsyn_4096.csv" --rows 4096 --seed "$SEED"
