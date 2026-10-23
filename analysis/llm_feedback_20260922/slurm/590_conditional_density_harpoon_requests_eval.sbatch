#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cond-harpoon-e
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${SAMPLE_JOB_ID:?SAMPLE_JOB_ID required}"

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
requests=(range category)
i=$SLURM_ARRAY_TASK_ID; request=${requests[$i]}
ours="$A/outputs/conditional_density_harpoon_${request}_${SAMPLE_JOB_ID}_${i}/samples"
proxy="$A/outputs/harpoon_mixed_proxy_${request}_1066201_${i}/samples"
dynamic="$A/outputs/harpoon_default_dynamic_${request}_1065643_${i}/samples"
extra=(
  --sample "tabdiff=$ours/tabdiff.csv"
  --sample "final_repair=$ours/calibrated_final_repair.csv"
  --sample "conditional_density=$ours/conditional_density_repaint.csv"
  --sample "conditional_density_plus_repair=$ours/conditional_density_repaint_plus_repair.csv"
  --sample "harpoon_proxy_weak=$proxy/proxy_n0.05_c1.csv"
  --sample "harpoon_proxy_strong=$proxy/proxy_n0.4_c8.csv"
  --sample "prior_dynamic_middle=$dynamic/xstate_reward_calibrated.csv"
)
if [[ "$request" == category ]]; then
  cdd_full="$A/outputs/cdd_matched_category_1069863/samples"
  cdd_delay="$A/outputs/cdd_matched_delayed_1069865/samples"
  extra+=(--sample "cdd_full=$cdd_full/cdd_full.csv" --sample "cdd_delayed=$cdd_delay/cdd_delayed.csv")
fi

cd "$REPO"
"$PY" "$A/scripts/evaluate_harpoon_constraint_suite.py" --repo "$REPO" \
  --request "$request" "${extra[@]}" --seed 20261376 \
  --output "$A/outputs/conditional_density_harpoon_eval_${request}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
