#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-strong-eval
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

: "${PARENT_JOB_ID:?PARENT_JOB_ID required}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
i=$SLURM_ARRAY_TASK_ID
requests=(category both)
request=${requests[$i]}
root="$A/outputs/harpoon_proxy_strong_${request}_${PARENT_JOB_ID}_${i}/samples"
if [[ "$request" == category ]]; then
  dynamic="$A/outputs/harpoon_default_dynamic_category_1065643_1/samples/xstate_reward_calibrated.csv"
else
  dynamic="$A/outputs/harpoon_reactivate_both_b2_1065811_3/samples/xstate_reward_calibrated.csv"
fi

"$PY" "$A/scripts/evaluate_harpoon_constraint_suite.py" --repo "$REPO" --request "$request" \
  --sample "unconditional=$root/unconditional.csv" \
  --sample "dynamic=$dynamic" \
  --sample "proxy_n0.4_c16=$root/proxy_n0.4_c16.csv" \
  --sample "proxy_n0.4_c32=$root/proxy_n0.4_c32.csv" \
  --sample "proxy_n0.4_c64=$root/proxy_n0.4_c64.csv" \
  --sample "proxy_n0.8_c32=$root/proxy_n0.8_c32.csv" \
  --certify-sample "proxy_n0.4_c16_certified=$root/proxy_n0.4_c16.csv" \
  --certify-sample "proxy_n0.4_c32_certified=$root/proxy_n0.4_c32.csv" \
  --certify-sample "proxy_n0.4_c64_certified=$root/proxy_n0.4_c64.csv" \
  --certify-sample "proxy_n0.8_c32_certified=$root/proxy_n0.8_c32.csv" \
  --output "$A/outputs/harpoon_proxy_strong_eval_${request}_${SLURM_ARRAY_JOB_ID}_${i}"
