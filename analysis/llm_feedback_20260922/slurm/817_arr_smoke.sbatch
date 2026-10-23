#!/usr/bin/env bash
# CPU-only import and contract checks for adaptive-resolution routing.
#SBATCH --job-name=llmtd-arr-smoke
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
set +x

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
export PYTHONPATH="$A/scripts:$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

"$PY" -m py_compile \
  "$A/scripts/adaptive_resolution_quantile_transport.py" \
  "$A/scripts/dense_numeric_factor_messages_arr.py"

"$PY" - <<'PY'
import numpy as np
import pandas as pd

from adaptive_resolution_quantile_transport import (
    AdaptiveResolutionQuantileTransport,
)

rng = np.random.default_rng(20261260)
n = 768
target = rng.normal(size=n)
train = pd.DataFrame({
    "target": target.astype("float32"),
    "x1": (0.8 * target + rng.normal(scale=0.6, size=n)).astype("float32"),
    "x2": (-0.5 * target + rng.normal(scale=0.8, size=n)).astype("float32"),
})
frame = train.sample(256, random_state=17).reset_index(drop=True)

dynamic = AdaptiveResolutionQuantileTransport(
    train, ["target", "x1", "x2"], [], condition_column="target",
    temperature=.75, confidence_floor=.52, confidence_full=.68,
    freeze_membership=False, preserve_target=False,
    global_tail_reserve_fraction=.05,
    scope="all", mode="copula", seed=20261260,
)
first, changed_first, diag_first = dynamic.apply(frame, .05)
perturbed = first.copy()
perturbed["target"] = (
    perturbed["target"] + np.linspace(-.5, .5, len(perturbed))
).astype("float32")
second, changed_second, diag_second = dynamic.apply(perturbed, .05)

preserved = AdaptiveResolutionQuantileTransport(
    train, ["target", "x1", "x2"], [], condition_column="target",
    preserve_target=True, scope="all", mode="copula", seed=20261261,
)
preserved_frame, _, preserved_diag = preserved.apply(frame, .05)

assert first.shape == frame.shape == second.shape
assert changed_first.shape == changed_second.shape == (len(frame), 3)
assert np.isfinite(first.to_numpy(float)).all()
assert np.isfinite(second.to_numpy(float)).all()
assert 2.0 <= diag_second["transport_resolution_mean"] <= 8.0
assert 0.0 <= diag_second["transport_resolution_switch_fraction"] <= 1.0
assert diag_first["transport_tail_rows_preserved"] > 0
weight_sum = sum(
    diag_first[f"transport_adaptive_resolution_q{k}_weight_mean"]
    for k in (2, 4, 8)
)
assert abs(weight_sum - 1.0) < 1e-8
assert np.allclose(preserved_frame["target"], frame["target"], equal_nan=True)
assert preserved_diag["transport_condition_column_preserved"] == 1
print("ARR_SMOKE_OK")
print({
    "expected_bins": diag_second["transport_resolution_mean"],
    "switch_fraction": diag_second["transport_resolution_switch_fraction"],
    "weight_l1_drift": diag_second[
        "transport_adaptive_resolution_weight_l1_drift"
    ],
    "changed_rows": int(changed_second.any(axis=1).sum()),
})
PY
