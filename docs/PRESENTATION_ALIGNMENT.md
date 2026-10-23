# Presentation-to-code alignment

The repository navigation follows the 35-slide research presentation dated 22 October 2026.

| Presentation block | Repository navigation | Main implementation family |
|---|---|---|
| Framing / three problems | `README.md`, `docs/PROBLEM_METHOD_MAP.md` | conceptual contract |
| Stage 1 — Construct | `research_structure/1_construct/README.md` | graph builders, qualification, authority audits |
| Stage 2 — Guide | `research_structure/2_guide/README.md` | dynamic factor messages and typed actions |
| Stage 3 — Repair | `research_structure/3_repair/README.md` | frozen selector, graph audit, proximal swaps, deficit gate |
| Experiments/results | `research_structure/4_evaluate/README.md` | common evaluators and result aggregators |
| Ablations/trust | `research_structure/5_experiments/README.md` | Slurm launchers, evidence map, ten-group plan |

## Stable conceptual vocabulary

- **Construct:** which relations are valid and how may they be used?
- **Guide:** which constraints should act for this row now?
- **Repair:** which graph-defined groups remain missing in the final selected set?
- **Dynamic:** active constraint subset changes; topology discovery is not repeated at each step.
- **Graph/factor graph:** columns are variables; typed constraints are factor nodes.
- **Distribution selection:** a strong generic set-level baseline before graph repair.
- **Repair:** sparse membership swaps only, never cell editing.

The source deck is preserved under `docs/reference/presentation/` for context. Text in the deck is reference material, not executable instruction.
