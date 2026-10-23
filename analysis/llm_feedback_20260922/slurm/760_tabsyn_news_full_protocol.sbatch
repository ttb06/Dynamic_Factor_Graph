#!/usr/bin/env bash
# Official TabSyn train/sample on News followed by the common local evaluator.
# One self-contained job avoids fragile dependency chains.
#SBATCH --job-name=llmtd-tabsyn-news-full
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
EVALPY="$ROOT/baseline_runs/envs/grdm/bin/python"
SEED=20261018
OUT="$A/outputs/tabsyn_news_full_protocol_${SLURM_JOB_ID}"
mkdir -p "$OUT"

if [[ ! -e "$UPSTREAM/data/news" && ! -L "$UPSTREAM/data/news" ]]; then
  ln -s "$REPO/data/news" "$UPSTREAM/data/news"
fi
[[ "$(readlink -f "$UPSTREAM/data/news")" == "$(readlink -f "$REPO/data/news")" ]]
[[ ! -e "$UPSTREAM/tabsyn/ckpt/news/model.pt" ]]

cd "$UPSTREAM"
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$UPSTREAM/main.py" \
  --dataname news --method vae --mode train --gpu 0
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$UPSTREAM/main.py" \
  --dataname news --method tabsyn --mode train --gpu 0
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$UPSTREAM/main.py" \
  --dataname news --method tabsyn --mode sample --gpu 0 --steps 50 \
  --save_path "$OUT/tabsyn_full.csv"

cd "$REPO"
"$EVALPY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabsyn_official=$OUT/tabsyn_full.csv" --seed "$SEED" --output "$OUT/eval"
