#!/usr/bin/env bash
#SBATCH --job-name=llmtd-strength-pair
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
SEEDS=(20261082 20261083); seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}
for scale in 1 4; do
  "$PY" "$A/scripts/rowwise_reverse_action_guidance.py" --repo "$REPO" --dataset default \
    --graph "$A/outputs/cross_dataset_catalog_default_1058707_1/constraint_catalog.json" \
    --schedule "$A/outputs/cross_dataset_trajectory_default_1058708_1/typed_readability_schedule.csv" \
    --config "$A/outputs/full_train_default_1033760/config.pkl" \
    --checkpoint "$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt" \
    --samples 8192 --steps 50 --seed "$seed" --min-support 24 --drift .10 \
    --density-weight 1.0 --edit-weight .5 --max-actions-per-row-edge 1 --cooldown-steps 3 \
    --schedule-mode common --common-cutoff .50 --gate-mode current --topology-mode dynamic \
    --state-gate stable5 --active-min-t .25 --active-max-t .50 \
    --numeric-guidance-mode constant --numeric-guidance-scale "$scale" \
    --online-modes x0_reward_calibrated \
    --output "$A/outputs/default_strength_exact_pair_s${seed}_g${scale}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
done

first="$A/outputs/default_strength_exact_pair_s${seed}_g1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/samples/unconditional.csv"
second="$A/outputs/default_strength_exact_pair_s${seed}_g4_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/samples/unconditional.csv"
test "$(sha256sum "$first" | cut -d' ' -f1)" = "$(sha256sum "$second" | cut -d' ' -f1)"
