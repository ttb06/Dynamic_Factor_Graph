#!/usr/bin/env bash
# Train-only row support controller over the two best official-sampler variants.
#SBATCH --job-name=llmtd-news-off-rollback
#SBATCH --partition=cpu
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"; i=$SLURM_ARRAY_TASK_ID
if (( i % 2 == 0 )); then
  SOURCE="$A/outputs/news_official_dynamic_1093963_3_s200"; LABEL=predictor_s200
else
  SOURCE="$A/outputs/news_official_heun_guidance_1093965_2_s100"; LABEL=heun_s100
fi
if (( i < 2 )); then SLACK=0; SLABEL=strict; else SLACK=.002; SLABEL=slack002; fi
OUT="$A/outputs/news_official_manifold_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}_${SLABEL}"
mkdir -p "$OUT/samples"; cd "$REPO"
"$PY" "$A/scripts/paired_manifold_rollback.py" \
  --train "$REPO/synthetic/news/real.csv" \
  --reference "$SOURCE/samples/final_repair.csv" \
  --candidate "$SOURCE/samples/dense_factor_then_certifier.csv" \
  --output "$OUT/samples/manifold_rollback.csv" \
  --prototypes 256 --components 20 --relative-slack "$SLACK" --seed 20261609
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$SOURCE/samples/unconditional.csv" \
  --sample "final_repair=$SOURCE/samples/final_repair.csv" \
  --sample "guided_cert=$SOURCE/samples/dense_factor_then_certifier.csv" \
  --sample "manifold_rollback=$OUT/samples/manifold_rollback.csv" \
  --seed 20261609 --output "$OUT/eval"
