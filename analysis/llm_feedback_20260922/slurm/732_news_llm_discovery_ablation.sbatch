#!/usr/bin/env bash
# Fair graph-discovery ablation against the same frozen 14-edge documented
# catalog: statistics only versus a second-model LLM proposal/review loop.
#SBATCH --job-name=llmtd-news-discovery-ablate
#SBATCH --partition=cpu
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
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
  LABEL=statistics_only
  EXTRA=()
else
  LABEL=terra_propose_review
  EXTRA=(--env-file "$ROOT/.env" --llm-propose --llm-review --model gpt-5.6-terra --max-output-tokens 10000)
fi
OUT="$A/outputs/news_discovery_ablation_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"

cd "$REPO"
"$PY" "$A/scripts/meaningful_typed_constraint_discovery.py" \
  --repo "$REPO" --dataset news --output "$OUT" \
  --max-candidates 24 --seed 20261017 \
  --documented-graph "$A/outputs/news_documented_catalog_v2_20261017/graph.json" \
  --old-empirical-graph "$A/outputs/news_semantic_program_1085296/empirical_graph.json" \
  "${EXTRA[@]}"
