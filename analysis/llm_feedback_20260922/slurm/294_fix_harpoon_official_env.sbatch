#!/usr/bin/env bash
#SBATCH --job-name=llmtd-harpoon-fix
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ENV=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/harpoon-official

# Opacus 1.5.4 imports torch.nn.RMSNorm, which is unavailable in the
# author-compatible PyTorch 2.2.2 environment.  This isolated baseline env is
# pinned to the pre-RMSNorm Opacus release; no repository code is changed.
"$ENV/bin/pip" install --no-deps --force-reinstall 'opacus==1.4.0'
"$ENV/bin/python" - <<'PY'
import torch
import opacus
from synthcity.metrics.eval_statistical import AlphaPrecision, KolmogorovSmirnovTest
print({"torch": torch.__version__, "opacus": opacus.__version__, "synthcity_metric_import": True})
PY
