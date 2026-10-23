#!/usr/bin/env bash
# CPU-only CFRW dependency/invariant smoke. No training or sampling on login.
#SBATCH --job-name=llmtd-cfrw-smoke
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x-%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x-%j.err
set -euo pipefail
set +x

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SCRIPTS="$A/scripts"

"$PY" -m py_compile \
  "$SCRIPTS/factorized_residual_witness.py" \
  "$SCRIPTS/dense_numeric_factor_messages_cfrw.py" \
  "$SCRIPTS/select_cfrw_candidates.py"

PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<'PY'
import numpy as np
import pandas as pd

from factorized_residual_witness import FactorizedResidualWitnessTransport

rng = np.random.default_rng(20261021)
n = 500
target = np.where(rng.random(n) > 0.5, "yes", "no")
target_shift = (target == "yes").astype(float)
x1 = 2.0 * target_shift + rng.normal(0.0, 0.35, n)
x2 = 0.85 * x1 + rng.normal(0.0, 0.20, n)
x3 = -0.55 * x2 + rng.normal(0.0, 0.25, n)
train = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": target})
preview = train.copy()
preview["x1"] = preview["x1"] * 1.25 + 0.35
preview["x2"] = preview["x2"] * 0.70 - 0.20
preview["x3"] = preview["x3"] + rng.normal(0.0, 0.30, n)
edges = [
    {"columns": ["x1", "x2"]},
    {"columns": ["x2", "x3"]},
]

operator = FactorizedResidualWitnessTransport(
    train,
    ["x1", "x2", "x3"],
    edges,
    mode="marginal_target_pair",
    target_column="target",
    trees=16,
    min_r2=0.01,
    max_pair_heads=4,
    top_factors_per_row=1,
    top_rows_per_factor=0.20,
    min_score=0.02,
    clip=2.0,
    seed=20261021,
)
first, first_mask, first_diag = operator.apply(preview, 0.10)
second, second_mask, second_diag = operator.apply(first, 0.10)

assert first_mask.shape == (n, 3)
assert second_mask.shape == (n, 3)
assert np.isfinite(first[["x1", "x2", "x3"]].to_numpy()).all()
assert preview["target"].equals(first["target"])
assert first["target"].equals(second["target"])
metadata = operator.metadata()
kinds = [head["kind"] for head in metadata["heads"]]
assert "marginal" in kinds and "target" in kinds and "pair" in kinds
assert metadata["heldout_used_for_fitting"] is False
assert first_diag["residual_witness_target_preserved"] == 1
assert first_diag["residual_witness_selected_cells"] > 0
assert second_diag["residual_witness_head_overlap"] >= 0.0
assert second_diag["residual_witness_head_switch_fraction"] >= 0.0
print("CFRW_SMOKE_OK", len(metadata["heads"]), int(first_mask.sum()), int(second_mask.sum()))
PY

PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$SCRIPTS/dense_numeric_factor_messages_cfrw.py" --help \
  > /dev/null

printf 'CFRW_SMOKE_OK\n'
