# Stage 3 — Repair

Question: **After selecting realistic rows, which graph-defined groups remain missing or distorted?**

## Pipeline

1. Generate an oversized candidate pool.
2. Use the frozen generic distribution selector (C3) as a strong anchor.
3. Audit factor-conditioned relation and coverage gaps using the Stage-1 graph semantics.
4. Activate repair only when graph deficit exceeds a train-only threshold.
5. Swap a very small number of selected/unselected rows.
6. Accept quality-first candidates subject to hard-factor and generic-quality tolerances.

Stage 3 never edits cell values and never reruns diffusion.

## Main implementation families

- `balanced_particle_resampling.py` — frozen generic selector.
- `graph_consistent_particle_resampling.py` — scalar graph-aware selection.
- `graph_pareto_anchored_resampling.py` — frozen-anchor Pareto control.
- `graph_proximal_swap_repair.py` — sparse quality-first membership repair.
- `audit_graph_consistency.py` — shared graph relation/coverage audit.

## Core novelty

The graph is reused across decision scales: trajectory intervention in Stage 2, set membership diagnostics/repair in Stage 3. Deficit gating makes the graph layer abstain when the generic anchor is already sufficient.
