#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-quality
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
harpoon="$A/outputs/harpoon_natural_graph_full_v3_s${seed}_20261010"
selected="$A/outputs/harpoon_natural_graph_aggregate_v3_20261010/selected_configs.json"
hard_label=$(jq -r '.hard.method' "$selected")
calibrated_label=$(jq -r '.calibrated.method' "$selected")

if [[ "$i" -eq 0 ]]; then
  dynamic="$A/outputs/natural_dag_stable_dynamic_1066791_0"
  frozen="$A/outputs/natural_dag_stable_frozen_1066791_1"
else
  base=$((3 * (i - 1)))
  dynamic="$A/outputs/natural_dag_stable_confirm_s${seed}_dynamic_1066822_${base}"
  frozen="$A/outputs/natural_dag_stable_confirm_s${seed}_frozen_1066822_$((base + 1))"
fi

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --sample "tabdiff=$dynamic/samples/unconditional.csv" \
  --sample "dynamic_graph=$dynamic/samples/xstate_reward_calibrated.csv" \
  --sample "frozen_graph=$frozen/samples/xstate_reward_calibrated.csv" \
  --sample "final_repair=$dynamic/samples/final_reward_calibrated.csv" \
  --sample "harpoon_hard=$harpoon/samples/${hard_label}.csv" \
  --sample "harpoon_calibrated=$harpoon/samples/${calibrated_label}.csv" \
  --seed "$seed" --output "$A/outputs/harpoon_natural_quality_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
