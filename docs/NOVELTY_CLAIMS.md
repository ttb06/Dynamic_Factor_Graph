# Novelty and claim map

This file separates implemented contributions from attractive future ideas. An agent must not promote a proposed extension into the paper's result section without a linked experiment.

## A. Primary paper story

**One dynamic constraint graph connects construction, trajectory guidance, and output repair.** The graph's semantics stay fixed, while its active factors and its decision role change.

### N1 — Authority-aware executable constraint graph

- Stores endpoints, conditions, support, reliability, authority, and permitted actions.
- Prevents statistical tendencies from being silently treated as universal laws.
- Position as infrastructure enabling the method, not necessarily a new mining algorithm.

### N2 — Row × diffusion-time × state activation

- Active factor = applicable × readable × needed × safe.
- Different rows can activate different factors at the same step.
- The same row can wait, act, then stop across the reverse trajectory.
- Strongest novelty candidate for Stage 2.

### N3 — Typed intervention routing

- Projection for verified hard numerical relations.
- Transport/messages for distributional numerical relations.
- Logit messages for categorical relations.
- Novelty comes from graph-controlled selective routing and budgeting, not from claiming each primitive operator is new.

### N4 — Cross-scale graph reuse

- During diffusion: decide whether a relation should act on this row now.
- At output: decide whether the relation remains represented across selected rows.
- Same endpoints, applicability, relation type, and authority; different actions.
- This is the cleanest unifying claim connecting sampling and selection.

### N5 — Deficit-gated proximal membership repair

- Anchor at a strong frozen distribution selector.
- Audit factor-conditioned relation/coverage deficits using train-only targets.
- Swap very few selected/unselected rows with quality-first acceptance.
- Abstain if the anchor is already consistent.
- Position as a safe graph-specific correction, not a replacement for generic distribution selection.

## B. Evidence status

- **Comprehensive:** Adult full method, fresh three-seed confirmation, strongest direct final case.
- **Coherent transfer:** Shoppers graph alignment improves, while generic fidelity can favor selection-only.
- **Boundary:** Magic and Default show graph gain does not imply universal external-quality gain.
- **Selection-only support:** Diabetes validates generic selection, not graph repair.
- **Trajectory support:** News and historical guidance experiments motivate timing/actions but use a separate protocol.

## C. Optional extensions worth exploring

These are intentionally labeled **proposed / unverified**.

1. **Counterfactual edge value** — estimate one-step validity–fidelity benefit before activating a factor.
2. **Temporal edge lifecycle** — model onset, persistence, hysteresis, and survival rather than independent gates.
3. **Uncertainty-priced budgets** — spend more intervention budget only when factor authority is high and preview uncertainty is low.
4. **Graph residual witness** — aggregate unresolved trajectory residuals and use them to prioritize Stage-3 swaps.
5. **Trajectory-to-set credit assignment** — measure which active edges caused final subgroup improvements or regressions.
6. **Constraint portfolio control** — select a compatible factor subset under conflict and total movement budgets.
7. **Graph trace explanations** — export row-level activation/action histories for audit, debugging, and qualitative analysis.
8. **Distributionally robust factor targets** — use uncertainty sets for low-support subgroup targets instead of single empirical estimates.

## D. Claims to avoid

- A new graph topology is learned at every diffusion step.
- Every semantic or LLM-proposed relation is a verified constraint.
- Graph repair universally outperforms generic selection.
- Local evaluator records equal paper-reported global SOTA.
- Internal graph loss alone is sufficient evidence of realism, utility, privacy, or fairness.
