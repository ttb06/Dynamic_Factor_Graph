# Codebase status

## What is stable

- Dynamic row/time factor activation and mixed-type actions.
- Generic C3/particle selection implementations.
- Graph-consistent, Pareto-anchored, and proximal selectors.
- Quality-first graph-proximal repair.
- Graph-deficit activation/no-op behavior.
- Shared evaluator and graph-audit pipeline.
- Slurm-only execution convention and provenance recording.

## What is still draft

- Script naming reflects experiment chronology rather than a public API.
- Some research scripts share helpers by direct import from the analysis directory.
- Final paper configs have not been reduced to one canonical YAML/JSON schema.
- Several historical launchers remain because they document negative results.
- Main baseline tables have uneven seed counts.
- Runtime, peak-memory, confidence-interval, and fairness analyses are incomplete.

## Do not silently change

- Do not delete negative-result scripts or summaries.
- Do not replace frozen C3 with a weaker random selector.
- Do not use held-out/test metrics inside selection.
- Do not relabel adapted/proxy baselines as official implementations.
- Do not mix paper-reported metrics with local evaluator rankings.
- Preserve datasets, checkpoints, and prior output roots on the server.

## Recommended next refactor

After the final experimental protocol is frozen, extract reusable modules into:

```text
research_graph/
  graph_schema.py
  activation.py
  actions.py
  c3.py
  proximal_repair.py
  audit.py
  evaluation.py
```

Do this only after reproducing the current final Adult result from the snapshot. Prematurely moving code risks invalidating provenance-linked launchers.

