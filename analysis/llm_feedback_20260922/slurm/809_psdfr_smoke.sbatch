#!/usr/bin/env bash
# CPU-only import and synthetic invariants for posterior-smoothed routing.
#SBATCH --job-name=llmtd-psdfr-smoke
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
export PYTHONPATH="$A/scripts:$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

"$PY" -m py_compile \
  "$A/scripts/posterior_quantile_transport.py" \
  "$A/scripts/dense_numeric_factor_messages_psdfr.py"

"$PY" - <<'PY'
import numpy as np
import pandas as pd

from posterior_quantile_transport import (
    PosteriorQuantileConditionalBatchDistributionTransport,
)

rng = np.random.default_rng(20261240)
n = 512
target = rng.normal(size=n)
train = pd.DataFrame({
    "target": target,
    "x1": 0.8 * target + rng.normal(scale=0.6, size=n),
    "x2": -0.5 * target + rng.normal(scale=0.8, size=n),
})
frame = train.sample(256, random_state=17).reset_index(drop=True)

dynamic = PosteriorQuantileConditionalBatchDistributionTransport(
    train, ["target", "x1", "x2"], [], condition_column="target",
    quantile_bins=4, temperature=0.75, freeze_membership=False,
    scope="all", mode="copula", seed=20261240,
)
first, changed_first, diag_first = dynamic.apply(frame, 0.05)
perturbed = first.copy()
perturbed["target"] = perturbed["target"] + np.linspace(-0.4, 0.4, len(perturbed))
second, changed_second, diag_second = dynamic.apply(perturbed, 0.05)

frozen = PosteriorQuantileConditionalBatchDistributionTransport(
    train, ["target", "x1", "x2"], [], condition_column="target",
    quantile_bins=4, temperature=0.75, freeze_membership=True,
    scope="all", mode="copula", seed=20261240,
)
frozen.apply(frame, 0.05)
_, _, frozen_second = frozen.apply(perturbed, 0.05)

assert first.shape == frame.shape == second.shape
assert changed_first.shape == changed_second.shape == (len(frame), 3)
assert np.isfinite(first.to_numpy(float)).all()
assert np.isfinite(second.to_numpy(float)).all()
assert 0.0 <= diag_first["transport_posterior_entropy_mean"] <= 1.0
assert 0.0 <= diag_second["transport_bin_switch_fraction"] <= 1.0
assert frozen_second["transport_bin_switch_fraction"] == 0.0
assert diag_second["transport_condition_call_index"] == 2
print("PSDFR_SMOKE_OK")
print({
    "dynamic_switch_fraction": diag_second["transport_bin_switch_fraction"],
    "entropy": diag_second["transport_posterior_entropy_mean"],
    "dynamic_changed_rows": int(changed_second.any(axis=1).sum()),
})
PY
