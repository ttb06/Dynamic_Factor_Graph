#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-sample3
#SBATCH --partition=prioritized
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -uo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
ENV="$ROOT/baseline_runs/envs/harpoon-official"
RUN_ROOT="$ROOT/baseline_runs/harpoon_official_multidata_20261008"
DATASETS=(adult default shoppers)
dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
NVLIB=$(find "$ENV/lib/python3.11/site-packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH="$ROOT/baseline_runs/envs/grdm/lib:$NVLIB:${LD_LIBRARY_PATH:-}"
cd "$RUN_ROOT/$dataset"

# One trial is an initial official-code rerun.  The paper reports five trials;
# only expand after this compatibility run completes on all three datasets.
status=0
for constraint in range category both or; do
  echo "BEGIN dataset=$dataset method=Harpoon constraint=$constraint"
  if "$ENV/bin/python" sampling_harpoon_ohe_tubular_generalconstraints.py \
      --dataname "$dataset" --constraint "$constraint" --num_trials 1 --gpu 0; then
    code=0
  else
    code=$?; status=1
  fi
  echo "END dataset=$dataset method=Harpoon constraint=$constraint exit=$code"

  echo "BEGIN dataset=$dataset method=RePaint constraint=$constraint"
  if "$ENV/bin/python" sampling_repaint_generalconstraints.py \
      --dataname "$dataset" --constraint "$constraint" --num_trials 1 --gpu 0; then
    code=0
  else
    code=$?; status=1
  fi
  echo "END dataset=$dataset method=RePaint constraint=$constraint exit=$code"
done
echo "aggregate_status=$status"
# Preserve successful rows even if an individual author command fails.
exit 0
