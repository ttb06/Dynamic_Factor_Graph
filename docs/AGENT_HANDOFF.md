# Agent handoff guide

## Read first

1. `README.md`
2. `research_structure/README.md`
3. `docs/PRESENTATION_ALIGNMENT.md`
4. `docs/PROBLEM_METHOD_MAP.md`
5. `docs/NOVELTY_CLAIMS.md`
6. `docs/CODE_WALKTHROUGH.md`
7. `docs/RESULTS_QUICK_REFERENCE.md`
8. `docs/PAPER_READY_RESEARCH_REPORT_20261022_VI.md`

## Operational constraints

- Remote host is orchestrated through SSH and Slurm.
- Do not run model/data/evaluation/packaging work on the login node.
- Do not run persistent agents on the server.
- Preserve user changes in `tabdiff/main.py`, `tabdiff/trainer.py`, and the analysis directory.
- Use unique job-scoped output roots.
- Record job IDs, seeds, source hashes, GPU metadata, and exact commands.
- Keep `.env`, API keys, datasets, checkpoints, and generated sample pools out of Git.

## Scientific constraints

- The test evaluator must not influence selection.
- A self-discovered factor is not automatically a true constraint.
- Use `B/G/R/old/final` controls.
- Keep negative results and predeclared gates.
- “Local SOTA” must be qualified as same-repository split/evaluator.
- Do not claim universal cross-dataset graph-repair gains.
- Never present items labeled proposed/unverified in `NOVELTY_CLAIMS.md` as completed contributions.
- Use the terms Construct, Guide, and Repair consistently with the presentation-aligned façade.

## Current next task

Freeze one final protocol, then execute the ten ablation groups described in `docs/ABLATION_PLAN_10_GROUPS_20261022_VI.md`. Reuse pools for selector-only ablations and regenerate only when trajectory/action settings change.

## How to inspect provenance

Start from a summary `RESULTS.md`, follow each row's root in `per_seed.csv`, read `provenance.txt`, then inspect selection metadata and graph audit. The package includes an omitted-artifact manifest for large files that remain on the server.

