#!/usr/bin/env bash
# Official TabDDPM implementation distributed with the TabSyn repository.
#SBATCH --job-name=llmtd-tabddpm
#SBATCH --partition=prioritized
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
UPSTREAM="$ROOT/baseline_repos/tabsyn"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/lcats-official/bin/python"
EVALPY="$ROOT/baseline_runs/envs/grdm/bin/python"
DATASETS=(adult magic)
GRAPHS=("$A/outputs/extended_bscg_adult_1081004/graph.json" "$A/outputs/extended_bscg_magic_1081009/graph.json")
i=$SLURM_ARRAY_TASK_ID
DATASET=${DATASETS[$i]}
GRAPH=${GRAPHS[$i]}
SEED=$((20261090 + i))
OUT="$A/outputs/tabddpm_official_${DATASET}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$OUT"
if [[ ! -e "$UPSTREAM/data/$DATASET" && ! -L "$UPSTREAM/data/$DATASET" ]]; then
  ln -s "$REPO/data/$DATASET" "$UPSTREAM/data/$DATASET"
fi
cd "$UPSTREAM"
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$UPSTREAM/main.py" \
  --dataname "$DATASET" --method tabddpm --mode train --gpu 0
"$PY" "$A/scripts/seeded_python_entrypoint.py" --seed "$SEED" --script "$UPSTREAM/main.py" \
  --dataname "$DATASET" --method tabddpm --mode sample --gpu 0 --ddim --steps 50 \
  --save_path "$OUT/tabddpm_full.csv"
"$PY" "$A/scripts/subsample_baseline_csv.py" --input "$OUT/tabddpm_full.csv" \
  --output "$OUT/tabddpm_4096.csv" --rows 4096 --seed "$SEED"
cd "$REPO"
"$EVALPY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset "$DATASET" --graph "$GRAPH" \
  --sample "TabDDPM-official=$OUT/tabddpm_4096.csv" \
  --seed "$SEED" --max-rows 4096 --output "$OUT/eval"

