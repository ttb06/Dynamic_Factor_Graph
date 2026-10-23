#!/usr/bin/env bash
# CPU-only import and synthetic invariants for uncertainty-aware routing.
#SBATCH --job-name=llmtd-uar-smoke
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
  "$A/scripts/posterior_quantile_transport.py" \
  "$A/scripts/uncertainty_aware_quantile_transport.py" \
  "$A/scripts/dense_numeric_factor_messages_uar.py"

"$PY" - <<'PY'
import numpy as np
import pandas as pd

from uncertainty_aware_quantile_transport import (
    UncertaintyAwarePosteriorQuantileTransport,
)

rng = np.random.default_rng(20261250)
n = 512
target = rng.normal(size=n)
train = pd.DataFrame({
    "target": target.astype("float32"),
    "x1": (0.8 * target + rng.normal(scale=0.6, size=n)).astype("float32"),
    "x2": (-0.5 * target + rng.normal(scale=0.8, size=n)).astype("float32"),
})
frame = train.sample(256, random_state=17).reset_index(drop=True)

global_router = UncertaintyAwarePosteriorQuantileTransport(
    train, ["target", "x1", "x2"], [], condition_column="target",
    quantile_bins=4, temperature=0.75, freeze_membership=False,
    confidence_floor=0.55, uncertainty_fallback="global",
    confidence_full=0.75, fallback_scale=0.25,
    membership_hysteresis=0.0, switch_margin=0.15,
    preserve_condition_column=True, preserve_tail_fraction=0.05,
    scope="all", mode="copula", seed=20261250,
)
first, changed_first, diag_first = global_router.apply(frame, 0.05)
perturbed = first.copy()
perturbed["target"] = (
    perturbed["target"] + np.linspace(-0.4, 0.4, len(perturbed))
).astype("float32")
second, changed_second, diag_second = global_router.apply(perturbed, 0.05)

abstain_router = UncertaintyAwarePosteriorQuantileTransport(
    train, ["target", "x1", "x2"], [], condition_column="target",
    quantile_bins=4, temperature=0.75, freeze_membership=False,
    confidence_floor=0.65, uncertainty_fallback="abstain",
    membership_hysteresis=0.0, preserve_condition_column=False,
    scope="all", mode="copula", seed=20261251,
)
abstained, changed_abstain, diag_abstain = abstain_router.apply(frame, 0.05)

assert first.shape == frame.shape == second.shape == abstained.shape
assert changed_first.shape == changed_second.shape == changed_abstain.shape
assert np.isfinite(first.to_numpy(float)).all()
assert np.isfinite(second.to_numpy(float)).all()
assert np.isfinite(abstained.to_numpy(float)).all()
assert np.allclose(first["target"], frame["target"])
assert np.allclose(second["target"], perturbed["target"])
assert 0.0 <= diag_second["transport_confidence_gate_mean"] <= 1.0
assert 0.0 <= diag_second["transport_bin_switch_fraction"] <= 1.0
assert diag_second["transport_switch_margin"] == 0.15
assert diag_second["transport_fallback_scale"] == 0.25
assert diag_second["transport_uncertainty_fallback_global"] == 1
assert diag_abstain["transport_uncertainty_fallback_abstain"] == 1
assert diag_first["transport_uncertain_row_events"] > 0
assert diag_first["transport_tail_rows_preserved"] > 0
print("UAR_SMOKE_OK")
print({
    "dynamic_switch_fraction": diag_second["transport_bin_switch_fraction"],
    "membership_l1_drift": diag_second["transport_membership_l1_drift"],
    "confidence_gate_mean": diag_second["transport_confidence_gate_mean"],
    "global_changed_rows": int(changed_second.any(axis=1).sum()),
    "abstain_changed_rows": int(changed_abstain.any(axis=1).sum()),
})
PY
