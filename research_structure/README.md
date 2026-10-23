# Research structure — one graph, three stages

This directory is a navigation façade over the chronological research tree. It does not duplicate implementations or alter provenance-linked launchers.

```text
1_construct/    Which relations are valid, applicable, and permitted?
2_guide/        Which active constraints should act on this row now?
3_repair/       Which graph-defined groups remain missing in the final set?
4_evaluate/     Did validity improve without sacrificing quality/utility/privacy?
5_experiments/  Which Slurm jobs, controls, seeds, and ablations establish the claim?
```

The shared object is the typed constraint graph: endpoints, conditions, factor type, support, reliability, authority, and allowed actions. Stage 2 changes trajectory values/logits; Stage 3 changes set membership only.

Use `component_index.json` for machine-readable routing to the actual scripts.
