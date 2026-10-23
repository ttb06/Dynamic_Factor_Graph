#!/usr/bin/env bash
# CPU-only factorized RFF-MMD smoke.
#SBATCH --job-name=llmtd-frmmd-smoke
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:08:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x-%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x-%j.err
set -euo pipefail
set +x

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SCRIPTS="$A/scripts"

"$PY" -m py_compile \
  "$SCRIPTS/factorized_rff_mmd.py" \
  "$SCRIPTS/dense_numeric_factor_messages_frmmd.py" \
  "$SCRIPTS/select_frmmd_candidates.py"

PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<'PY'
import numpy as np
import pandas as pd
from factorized_rff_mmd import FactorizedRFFMMDTransport

rng = np.random.default_rng(20261350)
n = 512
target = np.where(rng.random(n) > 0.5, "yes", "no")
x1 = 1.5 * (target == "yes") + rng.normal(0.0, 0.35, n)
x2 = 0.8 * x1 + rng.normal(0.0, 0.20, n)
x3 = -0.5 * x2 + rng.normal(0.0, 0.25, n)
train = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": target})
preview = train.copy()
preview["x1"] = preview["x1"] * 1.25 + 0.25
preview["x2"] = preview["x2"] * 0.75 - 0.20
operator = FactorizedRFFMMDTransport(
    train,
    ["x1", "x2", "x3"],
    [{"columns": ["x1", "x2"]}, {"columns": ["x2", "x3"]}],
    mode="marginal_pair",
    target_column="target",
    features=32,
    max_pair_heads=2,
    top_factors_per_row=1,
    top_rows_per_factor=0.20,
    min_score=0.05,
    action_clip=0.50,
    seed=20261350,
)
output, changed, diagnostics = operator.apply(preview, 0.10)
assert changed.shape == (n, 3) and changed.any()
assert output["target"].equals(preview["target"])
assert diagnostics["factor_mmd_target_preserved"] == 1
assert diagnostics["factor_mmd_predicted_energy_decrease"] > 0
assert diagnostics["factor_mmd_energy_before"] > 0
kinds = [head["kind"] for head in operator.metadata()["heads"]]
assert "marginal" in kinds and "pair" in kinds
print("FRMMD_SMOKE_OK", len(kinds), int(changed.sum()))
PY

PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$SCRIPTS/dense_numeric_factor_messages_frmmd.py" --help \
  > /dev/null
printf 'FRMMD_SMOKE_OK\n'
