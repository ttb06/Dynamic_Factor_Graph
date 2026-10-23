#!/usr/bin/env bash
# Re-evaluate available public/local generators with the corrected converged
# detector, identical 4,096-row budget, split, graph and metric code.
#SBATCH --job-name=llmtd-baseline-reeval
#SBATCH --partition=prioritized
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
i=$SLURM_ARRAY_TASK_ID

if [[ "$i" == 0 ]]; then
  DATASET=default
  GRAPH="$A/outputs/guidance_confirm_default_baseline_s20261082_1060913_2/evaluation_graph.json"
  TABDIFF="$A/outputs/guidance_confirm_default_baseline_s20261082_1060913_2/samples/unconditional.csv"
  CTGAN="$A/outputs/sdv_default_ctgan_1061903_0/synthetic.csv"
  TVAE="$A/outputs/sdv_default_tvae_1061903_1/synthetic.csv"
  TABSYN="$A/outputs/tabsyn_official_default_1082492_0/tabsyn_4096.csv"
  FOREST="$A/outputs/forestdiffusion_default_official_k100_1082464_2/forest_diffusion.csv"
else
  DATASET=shoppers
  GRAPH="$A/outputs/guidance_confirm_shoppers_baseline_s20261082_1060913_0/evaluation_graph.json"
  TABDIFF="$A/outputs/guidance_confirm_shoppers_baseline_s20261082_1060913_0/samples/unconditional.csv"
  CTGAN="$A/outputs/sdv_shoppers_ctgan_1057684/synthetic.csv"
  TVAE="$A/outputs/sdv_shoppers_tvae_1057684/synthetic.csv"
  TABSYN="$A/outputs/tabsyn_official_shoppers_1082492_1/tabsyn_4096.csv"
  FOREST="$A/outputs/forestdiffusion_shoppers_official_k100_1085152_3/forest_diffusion.csv"
fi

OUT="$A/outputs/matched_generator_baseline_reeval_${DATASET}_${SLURM_ARRAY_JOB_ID}_${i}"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset "$DATASET" --graph "$GRAPH" --max-rows 4096 \
  --sample "TabDiff=$TABDIFF" --sample "CTGAN-SDV=$CTGAN" \
  --sample "TVAE-SDV=$TVAE" --sample "TabSyn-official=$TABSYN" \
  --sample "ForestDiffusion-official=$FOREST" \
  --seed 20261017 --output "$OUT"
