# Experiments and ablations

## Control chain

- `B`: frozen TabDiff backbone.
- `G`: dynamic graph guidance only.
- `R`: random subset from the same oversized pool.
- `old`: frozen generic distribution selector.
- `graph`: deficit-gated proximal graph repair.

Reuse exactly the same generated pool for selection-only ablations. Regenerate only when trajectory guidance/actions change.

## Presentation-aligned ablation blocks

1. Graph construction and authority.
2. Dynamic versus fixed activation.
3. Typed action family.
4. Timing, pulses, strength, and trust bounds.
5. Generic distribution selection controls.
6. Shared-graph output objective.
7. Proximal swap/search design.
8. Deficit gate and no-op behavior.
9. Robustness, subgroups, privacy, and independent constraints.
10. Pool ratio, swap budget, runtime, memory, and GPU profile.

See `docs/ABLATION_PLAN_10_GROUPS_20261022_VI.md`, `docs/EXPERIMENT_AND_EVIDENCE_MAP.md`, and numbered launchers under `analysis/llm_feedback_20260922/slurm/`.
