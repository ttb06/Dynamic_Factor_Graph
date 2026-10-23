#!/usr/bin/env bash
#SBATCH --job-name=llmtd-cutsetup
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
CUTS="$ROOT/external/cuts"
ENV="$ROOT/baseline_runs/envs/cuts"
BASE="$CUTS/experiment_data/customizable_synthesizer_experiments/adult/trained_models/base_models"
WORK="$CUTS/experiment_data/customizable_synthesizer_experiments/adult/trained_models/workloads"
if [ ! -x "$ENV/bin/python" ]; then
  "$ROOT/baseline_runs/envs/grdm/bin/python" -m venv --system-site-packages "$ENV"
fi
"$ENV/bin/python" -m pip install --disable-pip-version-check "numpy==1.26.4" "xgboost==2.1.4"
mkdir -p "$BASE" "$WORK"
curl -fL --retry 4 --retry-delay 5 -o "$BASE/base_model_adult_all_three_with_labels_residual_[100,200,200,200,261]_gumbel_2000_total_variation_15000_16_42_inf.pickle" "https://files.sri.inf.ethz.ch/cuts/programmable_synthesizer_experiments/adult/trained_models/base_models/base_model_adult_all_three_with_labels_residual_%5b100,200,200,200,261%5d_gumbel_2000_total_variation_15000_16_42_inf.pickle"
curl -fL --retry 4 --retry-delay 5 -o "$WORK/workloads_all_three_with_labels_42_inf.pickle" "https://files.sri.inf.ethz.ch/cuts/programmable_synthesizer_experiments/adult/trained_models/workloads/workloads_all_three_with_labels_42_inf.pickle"
cd "$CUTS"
"$ENV/bin/python" - <<'PY'
import numpy, torch, xgboost
from customizable_synthesizer import CuTS
print({'numpy': numpy.__version__, 'torch': torch.__version__, 'xgboost': xgboost.__version__, 'cuts_import': CuTS.__name__})
PY
