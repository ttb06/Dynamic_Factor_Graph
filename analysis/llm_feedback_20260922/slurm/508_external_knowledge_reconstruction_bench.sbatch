#!/usr/bin/env bash
#SBATCH --job-name=llmtd-extknowledge
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python

datasets=(default shoppers diabetes)
graphs=(
  "$A/outputs/graph_rate_expanded_multi_entry_v3_graph_rate_1067515_1/evaluation_graph.json"
  "$A/outputs/shoppers_promote_confirm_s20261383_dynamic_1067743_0/evaluation_graph.json"
  "$A/outputs/diabetes_expanded_confirm_s20261383_dynamic_1068457_0/evaluation_graph.json"
)
samples=(
  "$A/outputs/graph_rate_expanded_multi_entry_v3_graph_rate_1067515_1/samples/unconditional.csv"
  "$A/outputs/shoppers_promote_confirm_s20261383_dynamic_1067743_0/samples/unconditional.csv"
  "$A/outputs/diabetes_expanded_confirm_s20261383_dynamic_1068457_0/samples/unconditional.csv"
)
i=$SLURM_ARRAY_TASK_ID
dataset=${datasets[$i]}

"$PY" "$A/scripts/external_knowledge_reconstruction_bench.py" \
  --repo "$REPO" --dataset "$dataset" --graph "${graphs[$i]}" \
  --sample "${samples[$i]}" --min-support 30 --top-k 8 --sparse-context 4 \
  --passes 3 --seed $((20261012+i)) \
  --output "$A/outputs/external_knowledge_${dataset}_${SLURM_ARRAY_JOB_ID}_${i}"
