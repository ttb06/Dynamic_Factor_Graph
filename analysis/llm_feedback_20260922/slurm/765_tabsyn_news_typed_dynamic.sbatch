#!/usr/bin/env bash
# Cross-backbone transfer of the full 11-edge News typed graph.  Submitted
# with a delayed start rather than a Slurm dependency; the checkpoint existence
# assertion makes failure explicit if the upstream reproduction did not finish.
#SBATCH --job-name=llmtd-tabsyn-news-dyn
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
TABSYN="$ROOT/baseline_repos/tabsyn"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/lcats-official/bin/python"
EVALPY="$ROOT/baseline_runs/envs/grdm/bin/python"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
CKPT="$TABSYN/tabsyn/ckpt/news/model.pt"
i=$SLURM_ARRAY_TASK_ID
case "$i" in
  0) LABEL=s05; STRENGTH=.05;;
  1) LABEL=s10; STRENGTH=.10;;
  2) LABEL=s20; STRENGTH=.20;;
  3) LABEL=s40; STRENGTH=.40;;
esac
OUT="$A/outputs/tabsyn_news_typed_dynamic_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
[[ -s "$CKPT" ]]

cd "$REPO"
"$PY" "$A/scripts/tabsyn_typed_program_guidance.py" \
  --tabsyn-repo "$TABSYN" --dataset news --checkpoint "$CKPT" --graph "$GRAPH" \
  --samples 8192 --steps 50 --seed 20261020 --strength "$STRENGTH" \
  --q-min .10 --q-max .40 --schedule middle_peak --trust-radius .50 \
  --stability-threshold .001 --persistence 2 --guidance-stage both \
  --family-strengths llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5 \
  --family-transports statistical_order_program=rank --output "$OUT"
"$EVALPY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabsyn=$OUT/tabsyn_baseline.csv" \
  --sample "tabsyn_dynamic=$OUT/tabsyn_typed_dynamic.csv" \
  --seed 20261020 --skip-detection --output "$OUT/eval"

