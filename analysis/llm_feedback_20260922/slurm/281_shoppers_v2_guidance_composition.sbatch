#!/usr/bin/env bash
#SBATCH --job-name=llmtd-v2-guide-compose
#SBATCH --partition=prioritized
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
SEEDS=(20261090 20261091); seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}
schedule="$A/outputs/actionable_trajectory_v2_shoppers_1060954_0/typed_readability_schedule.csv"
evaluation="$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json"

for graph_kind in full portfolio; do
  graph="$A/outputs/actionable_catalog_v2_shoppers_1060953_0/constraint_catalog.json"
  if [[ "$graph_kind" == portfolio ]]; then
    graph="$A/outputs/trajectory_portfolio_v2_shoppers_1060979_0/constraint_catalog.json"
  fi
  for variant in c1 c4 hard4_near1 hard4_near2; do
    mode=constant; scale=1; hard=4; near=1
    if [[ "$variant" == c4 ]]; then scale=4; fi
    if [[ "$variant" == hard4_near1 ]]; then mode=hardness_adaptive; fi
    if [[ "$variant" == hard4_near2 ]]; then mode=hardness_adaptive; near=2; fi
    "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset shoppers \
      --graph "$graph" --evaluation-graph "$evaluation" --schedule "$schedule" \
      --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
      --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
      --samples 8192 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
      --density-weight 1.0 --edit-weight .5 --graph-weight 1 \
      --max-actions-per-row-edge 1 --cooldown-steps 3 --schedule-mode common --common-cutoff .50 \
      --gate-mode current --topology-mode dynamic --state-gate current --active-min-t .25 --active-max-t .50 \
      --numeric-guidance-mode "$mode" --numeric-guidance-scale "$scale" \
      --hard-guidance-scale "$hard" --near-guidance-scale "$near" --soft-guidance-scale .5 \
      --guided-final-repair --online-modes x0_reward_calibrated \
      --output "$A/outputs/shoppers_v2_guidance_${graph_kind}_${variant}_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  done
done

hash_count=$(sha256sum "$A"/outputs/shoppers_v2_guidance_*_s${seed}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/samples/unconditional.csv | cut -d' ' -f1 | sort -u | wc -l)
test "$hash_count" -eq 1
