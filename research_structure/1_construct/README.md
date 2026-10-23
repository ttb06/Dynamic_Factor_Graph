# Stage 1 — Construct

Question: **Which relations matter, where do they apply, and how may they be used?**

## Pipeline

1. Gather known rules, schema/metadata, train-only statistics, and optional offline semantic proposals.
2. Type-check endpoints and factor operators.
3. Measure support, counterexamples, conflicts, and leakage risk.
4. Assign authority: enforce, score-only, observe, or reject.
5. Export an executable graph consumed by guidance and output audit.

## Main implementation families

- `meaningful_typed_constraint_discovery.py`
- `qualify_typed_graph.py`
- `curate_typed_actuation_graph.py`
- `build_adult_cuts_typed_graph.py`
- `build_news_public_typed_graph.py`
- `compiler_typecheck_property_audit.py`
- `compiler_counterexample_audit.py`
- `audit_constraint_meaningfulness.py`

All paths above are under `analysis/llm_feedback_20260922/scripts/`.

## Claim boundary

The paper can claim an authority-aware executable graph representation and verification pipeline. It need not claim a universally new constraint-mining algorithm.
