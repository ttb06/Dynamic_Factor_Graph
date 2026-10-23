#!/usr/bin/env bash
# Official TabSyn on Magic.
#SBATCH --job-name=llmtd-magic-tabsyn
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
SEED=20261082
OUT="$A/outputs/tabsyn_official_magic_${SLURM_JOB_ID}"
mkdir -p "$OUT"
if [[ ! -e "$UPSTREAM/data/magic" && ! -L "$UPSTREAM/data/magic" ]]; then
  ln -s "$REPO/data/magic" "$UPSTREAM/data/magic"
fi
cd "$UPSTREAM"
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$UPSTREAM/main.py" \
  --dataname magic --method vae --mode train --gpu 0
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$UPSTREAM/main.py" \
  --dataname magic --method tabsyn --mode train --gpu 0
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$UPSTREAM/main.py" \
  --dataname magic --method tabsyn --mode sample --gpu 0 --steps 50 \
  --save_path "$OUT/tabsyn_full.csv"
"$PY" "$A/scripts/subsample_baseline_csv.py" --input "$OUT/tabsyn_full.csv" \
  --output "$OUT/tabsyn_4096.csv" --rows 4096 --seed "$SEED"
cd "$REPO"
"$EVALPY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset magic \
  --graph "$A/outputs/extended_bscg_magic_1081009/graph.json" \
  --sample "TabSyn-official=$OUT/tabsyn_4096.csv" \
  --seed "$SEED" --max-rows 4096 --output "$OUT/eval"

