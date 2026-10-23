#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-proxy-eval
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2
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
requests=(range category both)
request=${requests[$i]}
root="$A/outputs/harpoon_mixed_proxy_${request}_${PARENT_JOB_ID}_${i}/samples"
main_root="$A/outputs/harpoon_default_dynamic_${request}_1065643_${i}/samples"
extra=(--sample "unconditional=$root/unconditional.csv"
       --sample "final_repair=$main_root/final_reward_calibrated.csv"
       --sample "proxy_n0.05_c1=$root/proxy_n0.05_c1.csv"
       --sample "proxy_n0.1_c2=$root/proxy_n0.1_c2.csv"
       --sample "proxy_n0.2_c4=$root/proxy_n0.2_c4.csv"
       --sample "proxy_n0.4_c8=$root/proxy_n0.4_c8.csv")
if [[ "$request" == range ]]; then
  extra+=(--sample "dynamic_reactivation=$A/outputs/harpoon_reactivate_range_b2_1065811_0/samples/xstate_reward_calibrated.csv")
elif [[ "$request" == both ]]; then
  extra+=(--sample "dynamic_reactivation=$A/outputs/harpoon_reactivate_both_b2_1065811_3/samples/xstate_reward_calibrated.csv")
else
  extra+=(--sample "dynamic_middle=$main_root/xstate_reward_calibrated.csv")
fi

"$PY" "$A/scripts/evaluate_harpoon_constraint_suite.py" --repo "$REPO" \
  --request "$request" "${extra[@]}" \
  --output "$A/outputs/harpoon_mixed_proxy_eval_${request}_${SLURM_ARRAY_JOB_ID}_${i}"
