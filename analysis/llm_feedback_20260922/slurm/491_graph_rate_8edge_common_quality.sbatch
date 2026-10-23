#!/usr/bin/env bash
#SBATCH --job-name=llmtd-graph-rate-8edge-quality
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
local_run="$A/outputs/graph_rate_expanded_multi_entry_v3_severity_1067515_0"
graph_run="$A/outputs/graph_rate_expanded_multi_entry_v3_graph_rate_1067515_1"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$A/outputs/natural_dag_multi_entry_v3_20261010/graph.json" \
  --sample "continued_unguided=$local_run/samples/unconditional.csv" \
  --sample "split_median_local_8edge=$local_run/samples/xstate_reward_calibrated.csv" \
  --sample "split_median_final_8edge=$local_run/samples/final_reward_calibrated.csv" \
  --sample "split_median_graph_rate_8edge=$graph_run/samples/xstate_reward_calibrated.csv" \
  --seed 20261376 --output "$A/outputs/graph_rate_8edge_common_quality_${SLURM_JOB_ID}"
