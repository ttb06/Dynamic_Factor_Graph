#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mixgraph-quality
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

: "${EVAL_JOB_ID:?EVAL_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
pre="$A/outputs/natural_dag_stable_dynamic_1066791_0"
hcont="$A/outputs/mixed_graph_eval_head_continue_${EVAL_JOB_ID}_0"
hgraph="$A/outputs/mixed_graph_eval_head_calibrated_${EVAL_JOB_ID}_1"
fcont="$A/outputs/mixed_graph_eval_full_continue_${EVAL_JOB_ID}_2"
fgraph="$A/outputs/mixed_graph_eval_full_calibrated_${EVAL_JOB_ID}_3"

for path in "$hcont" "$hgraph" "$fcont" "$fgraph"; do
  test -s "$path/method_metrics.csv"
done

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$A/outputs/natural_dag_stable_v3_20261010/graph.json" \
  --sample "tabdiff=$pre/samples/unconditional.csv" \
  --sample "dynamic_pretrained=$pre/samples/xstate_reward_calibrated.csv" \
  --sample "head_continue=$hcont/samples/unconditional.csv" \
  --sample "head_continue_dynamic=$hcont/samples/xstate_reward_calibrated.csv" \
  --sample "head_calibrated=$hgraph/samples/unconditional.csv" \
  --sample "head_calibrated_dynamic=$hgraph/samples/xstate_reward_calibrated.csv" \
  --sample "full_continue=$fcont/samples/unconditional.csv" \
  --sample "full_continue_dynamic=$fcont/samples/xstate_reward_calibrated.csv" \
  --sample "full_calibrated=$fgraph/samples/unconditional.csv" \
  --sample "full_calibrated_dynamic=$fgraph/samples/xstate_reward_calibrated.csv" \
  --seed 20261376 --output "$A/outputs/mixed_graph_common_quality_${SLURM_JOB_ID}"
