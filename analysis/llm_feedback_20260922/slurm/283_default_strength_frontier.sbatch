#!/usr/bin/env bash
#SBATCH --job-name=llmtd-strength-frontier
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
seed=20261092
for scale in .5 1 1.5 2 3 4; do
  label=${scale/./p}
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
    --graph "$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json" \
    --schedule "$A/outputs/cross_dataset_trajectory_default_1058708_1/typed_readability_schedule.csv" \
    --config "$A/outputs/full_train_default_1033760/config.pkl" \
    --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
    --samples 8192 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --graph-weight 1 --max-actions-per-row-edge 1 --cooldown-steps 3 \
    --schedule-mode common --common-cutoff .50 --gate-mode current --topology-mode dynamic \
    --state-gate stable5 --active-min-t .25 --active-max-t .50 \
    --numeric-guidance-mode constant --numeric-guidance-scale "$scale" \
    --guided-final-repair --online-modes x0_reward_calibrated \
    --output "$A/outputs/default_strength_frontier_g${label}_${SLURM_JOB_ID}"
done
hash_count=$(sha256sum "$A"/outputs/default_strength_frontier_*_${SLURM_JOB_ID}/samples/unconditional.csv | cut -d' ' -f1 | sort -u | wc -l)
test "$hash_count" -eq 1
