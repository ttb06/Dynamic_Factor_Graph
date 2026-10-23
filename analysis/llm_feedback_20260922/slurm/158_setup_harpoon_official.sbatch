#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
SOURCE="$ROOT/baseline_repos/harpoon"
RUN="$ROOT/baseline_runs/harpoon_general_adult_20261002"
ENV="$ROOT/baseline_runs/envs/harpoon-official"

if [[ ! -d "$RUN" ]]; then
  mkdir -p "$RUN"
  cp -a "$SOURCE/." "$RUN/"
fi
if [[ ! -x "$ENV/bin/python" ]]; then
  "$ROOT/baseline_runs/envs/grdm/bin/python" -m venv --system-site-packages "$ENV"
fi
"$ENV/bin/python" -m pip install --upgrade "pip<25.3"
"$ENV/bin/python" -m pip install "synthcity==0.2.12" "category-encoders>=2.6,<3" "hyperimpute==0.1.17" "xgboost>=2.1,<3"
"$ENV/bin/python" - <<'PY'
import torch, sklearn, synthcity
print({"torch": torch.__version__, "sklearn": sklearn.__version__, "synthcity": synthcity.__version__})
PY
