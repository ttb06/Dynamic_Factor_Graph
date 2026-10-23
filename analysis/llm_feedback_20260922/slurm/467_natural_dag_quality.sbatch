#!/usr/bin/env bash
#SBATCH --job-name=llmtd-dag-quality
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID
seeds=(20261376 20261377 20261378)
seed=${seeds[$i]}

if [[ "$i" -eq 0 ]]; then
  dynamic="$A/outputs/natural_dag_stable_dynamic_1066791_0"
  frozen="$A/outputs/natural_dag_stable_frozen_1066791_1"
  late="$A/outputs/natural_dag_stable_late_only_1066791_2"
else
  base=$((3 * (i - 1)))
  dynamic="$A/outputs/natural_dag_stable_confirm_s${seed}_dynamic_1066822_${base}"
  frozen="$A/outputs/natural_dag_stable_confirm_s${seed}_frozen_1066822_$((base + 1))"
  late="$A/outputs/natural_dag_stable_confirm_s${seed}_late_only_1066822_$((base + 2))"
fi

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --sample "unconditional=$dynamic/samples/unconditional.csv" \
  --sample "dynamic=$dynamic/samples/xstate_reward_calibrated.csv" \
  --sample "frozen=$frozen/samples/xstate_reward_calibrated.csv" \
  --sample "late_only=$late/samples/xstate_reward_calibrated.csv" \
  --sample "final_only=$dynamic/samples/final_reward_calibrated.csv" \
  --seed "$seed" --output "$A/outputs/natural_dag_quality_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
