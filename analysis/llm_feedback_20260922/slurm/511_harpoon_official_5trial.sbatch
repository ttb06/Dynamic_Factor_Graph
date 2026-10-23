#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-off5
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
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
PY="$ROOT/baseline_runs/envs/harpoon-official/bin/python"
datasets=(adult default shoppers)
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}
source="$ROOT/baseline_runs/harpoon_official_multidata_20261008/$dataset"
out="$A/outputs/harpoon_official_5trial_${dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
scratch="${SLURM_TMPDIR:-/tmp/harpoon_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}/work"
mkdir -p "$scratch" "$out"
cp -a "$source/." "$scratch/"
cd "$scratch"
rm -f experiments/general_constraints_updated_utility.csv
for constraint in range category both or; do
  "$PY" sampling_harpoon_ohe_tubular_generalconstraints.py \
    --dataname "$dataset" --constraint "$constraint" --timesteps 200 --num_trials 5 --gpu 0
done
cp experiments/general_constraints_updated_utility.csv "$out/general_constraints_updated_utility.csv"
git -C "$ROOT/baseline_repos/harpoon" rev-parse HEAD > "$out/official_commit.txt"
