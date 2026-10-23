#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cdgm-news
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
# This venv inherits its Python standard library from the grdm conda prefix.
# Its ICU extension therefore needs the matching conda libstdc++ at runtime.
export LD_LIBRARY_PATH="$ROOT/baseline_runs/envs/grdm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
versions=(unconstrained constrained)
version=${versions[$SLURM_ARRAY_TASK_ID]}
out="$A/outputs/cdgm_news_official_${version}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
scratch="${SLURM_TMPDIR:-/tmp/cdgm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}/work"
mkdir -p "$scratch" "$out"
cp -a "$ROOT/baseline_repos/constrained_dgm/." "$scratch/"
cp "$TABDIFF/data/news/OnlineNewsPopularity/OnlineNewsPopularity.csv" "$scratch/data/news/news.csv"
cd "$scratch"
"$PY" - <<'PY'
from sklearn.model_selection import train_test_split
import pandas as pd
d = pd.read_csv('data/news/news.csv').drop_duplicates()
d = d.drop(columns=['url'])
d.columns = [c[1:] if c.startswith(' ') else c for c in d.columns]
train, rest = train_test_split(d, test_size=.2, random_state=1)
val, test = train_test_split(rest, test_size=(.10/.90), random_state=1)
train.to_csv('data/news/train_data.csv', index=False)
val.to_csv('data/news/val_data.csv', index=False)
test.to_csv('data/news/test_data.csv', index=False)
print({'train': train.shape, 'val': val.shape, 'test': test.shape})
PY
extra=()
if [[ "$version" == constrained ]]; then extra=(--version constrained --label_ordering corr); fi
"$PY" main_tvae.py news --seed 9 --epochs 50 --batch_size 70 --l2scale 0.00001 \
  --loss_factor 3 --wandb_mode disabled --skip_evaluation "${extra[@]}"
generated=$(find "outputs/TVAE_out/news/$version" -name generated_data.pkl -type f | head -n 1)
test -n "$generated"
cp "$generated" "$out/generated_data.pkl"
git rev-parse HEAD > "$out/official_commit.txt"
