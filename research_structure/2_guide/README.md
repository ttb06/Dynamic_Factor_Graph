# Stage 2 — Guide

Question: **Which constraint should act for this row at this reverse step?**

## Activation contract

```text
Active(edge, row, step) = Applicable × Readable × Needed × Safe
```

- Applicable: the factor's condition holds for the current row/state.
- Readable: the relevant clean-preview columns are sufficiently stable.
- Needed: the factor residual is large enough to justify action.
- Safe: authority, uncertainty, trust region, and movement budget permit action.

## Typed actions

- Projection: hard numerical feasibility.
- Numerical transport/messages: conditional or distributional relations.
- Categorical logit messages: still-masked categorical dependencies.

## Main implementation families

- `typed_dynamic_graph_guidance.py`
- `dense_numeric_factor_messages.py`
- `dense_numeric_factor_messages_arr.py`
- `categorical_logit_guidance.py`
- `dynamic_graph_logit_messages.py`
- `applicability_gate_trajectory.py`
- `temporal_contract_guidance.py`
- `dynamic_graph_conditional_repaint.py`
- `rowwise_reverse_action_guidance.py`

## Key controls

Fixed/global, row-only, time-only, row+time, no residual gate, no uncertainty/safety gate, early/middle/late, pulse/continuous, and matched action budget.
