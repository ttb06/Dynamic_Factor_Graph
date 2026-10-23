#!/usr/bin/env bash
# Official TabSyn train/sample/evaluate on the repository Adult split.
#SBATCH --job-name=llmtd-adult-tabsyn
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
SEED=20261049
OUT="$A/outputs/tabsyn_official_adult_${SLURM_JOB_ID}"
mkdir -p "$OUT"
if [[ ! -e "$UPSTREAM/data/adult" && ! -L "$UPSTREAM/data/adult" ]]; then
  ln -s "$REPO/data/adult" "$UPSTREAM/data/adult"
fi
[[ "$(readlink -f "$UPSTREAM/data/adult")" == "$(readlink -f "$REPO/data/adult")" ]]
cd "$UPSTREAM"
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$UPSTREAM/main.py" \
  --dataname adult --method vae --mode train --gpu 0
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$UPSTREAM/main.py" \
  --dataname adult --method tabsyn --mode train --gpu 0
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$UPSTREAM/main.py" \
  --dataname adult --method tabsyn --mode sample --gpu 0 --steps 50 \
  --save_path "$OUT/tabsyn_full.csv"
"$PY" "$A/scripts/subsample_baseline_csv.py" --input "$OUT/tabsyn_full.csv" \
  --output "$OUT/tabsyn_4096.csv" --rows 4096 --seed "$SEED"
cd "$REPO"
"$EVALPY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset adult \
  --graph "$A/outputs/extended_bscg_adult_1081004/graph.json" \
  --sample "TabSyn-official=$OUT/tabsyn_4096.csv" \
  --seed "$SEED" --max-rows 4096 --output "$OUT/eval"

