#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-prepare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
SOURCE="$ROOT/baseline_repos/harpoon"
RUN_ROOT="$ROOT/baseline_runs/harpoon_official_multidata_20261008"
ENV="$ROOT/baseline_runs/envs/harpoon-official"

# The first setup attempt installed the author dependencies but failed its
# final import because the packaged CUDA libraries were not on the loader path.
# Keep that isolated environment and only add the two lightweight Excel readers
# required by the author's Default preprocessing.
"$ENV/bin/python" -m pip install --quiet "xlrd==2.0.1" "openpyxl>=3.1,<4"

for dataset in adult default shoppers; do
  target="$RUN_ROOT/$dataset"
  if [[ ! -d "$target" ]]; then
    mkdir -p "$target"
    cp -a "$SOURCE/." "$target/"
    if [[ -f "$target/experiments/general_constraints_updated_utility.csv" ]]; then
      mv "$target/experiments/general_constraints_updated_utility.csv" \
         "$target/experiments/paper_reported_general_constraints_updated_utility.csv"
    fi
  fi
done

# Adult is already shipped in the author repository. Download and process only
# the two missing mixed-type datasets, preserving the author's 70/30 split and
# seed 1234 rather than substituting the TabDiff split.
for dataset in default shoppers; do
  cd "$RUN_ROOT/$dataset"
  "$ENV/bin/python" - "$dataset" <<'PY'
import sys
from download_and_process import download_from_uci, process_default, process_shoppers, train_test_split

name = sys.argv[1]
download_from_uci(name)
{"default": process_default, "shoppers": process_shoppers}[name]()
train_test_split(name, ratio=0.7)
PY
done

for dataset in adult default shoppers; do
  test -s "$RUN_ROOT/$dataset/datasets/$dataset/train.csv"
  test -s "$RUN_ROOT/$dataset/datasets/$dataset/test.csv"
done
