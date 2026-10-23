#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cdgm-match
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
TABDIFF="$ROOT/baseline_repos/tabdiff"
A="$TABDIFF/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/harpoon-official/bin/python"
export LD_LIBRARY_PATH="$ROOT/baseline_runs/envs/grdm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
versions=(unconstrained constrained)
version=${versions[$SLURM_ARRAY_TASK_ID]}
out="$A/outputs/cdgm_news_matched_${version}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
scratch="${SLURM_TMPDIR:-/tmp/cdgm_match_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}/work"
mkdir -p "$scratch" "$out"
cp -a "$ROOT/baseline_repos/constrained_dgm/." "$scratch/"
cd "$scratch"
"$PY" "$A/scripts/prepare_cdgm_news_matched_split.py" --tabdiff-repo "$TABDIFF" --output data/news
extra=()
if [[ "$version" == constrained ]]; then extra=(--version constrained --label_ordering corr); fi
"$PY" main_tvae.py news --seed 9 --epochs 50 --batch_size 70 --l2scale 0.00001 \
  --loss_factor 3 --wandb_mode disabled --skip_evaluation "${extra[@]}"
generated=$(find "outputs/TVAE_out/news/$version" -name generated_data.pkl -type f | head -n 1)
test -n "$generated"
cp "$generated" "$out/generated_data.pkl"
git rev-parse HEAD > "$out/official_commit.txt"
