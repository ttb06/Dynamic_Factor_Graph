#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-common
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
severities=(graph_rate_expanded_multi_entry_v3_severity_1067515_0 graph_rate_8edge_s20261377_severity_1067520_0 graph_rate_8edge_s20261378_severity_1067520_2)
graphs=(graph_rate_expanded_multi_entry_v3_graph_rate_1067515_1 graph_rate_8edge_s20261377_graph_rate_1067520_1 graph_rate_8edge_s20261378_graph_rate_1067520_3)
harpoons=(harpoon_default_8edge_confirm_s20261376_1067613_0 harpoon_default_8edge_confirm_s20261377_1067613_1 harpoon_default_8edge_confirm_s20261378_1067613_2)
seed=${seeds[$i]}
local_run="$A/outputs/${severities[$i]}"
graph_run="$A/outputs/${graphs[$i]}"
harpoon_run="$A/outputs/${harpoons[$i]}"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$A/outputs/natural_dag_multi_entry_v3_20261010/graph.json" \
  --sample "continued_unguided=$local_run/samples/unconditional.csv" \
  --sample "final_repair=$local_run/samples/final_reward_calibrated.csv" \
  --sample "dynamic_graph_rate=$graph_run/samples/xstate_reward_calibrated.csv" \
  --sample "harpoon_hard_w8=$harpoon_run/samples/harpoon_hard_w8.csv" \
  --seed "$seed" --output "$A/outputs/harpoon_dynamic_common_s${seed}_${SLURM_ARRAY_JOB_ID}_${i}"
