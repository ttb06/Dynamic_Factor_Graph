# Dynamic Constraint Graph for Tabular Diffusion — Research Snapshot (Draft)

This repository snapshot is intentionally **research-complete but not paper-final**. It preserves implementation code, Slurm launchers, metadata, summaries, and logs so a human or coding agent can understand and continue the project.

## One graph. Three stages.

Mixed-type tabular diffusion must apply the right constraint to the right row at the right reverse step, while keeping the final table realistic and representative. This project uses one typed constraint graph across three stages:

1. **Construct** — discover, verify, type, and authorize usable constraints.
2. **Guide** — activate a row-, timestep-, and state-specific subgraph and apply bounded typed actions.
3. **Repair** — reuse the same graph semantics to detect factor-conditioned set mismatch and make sparse membership swaps.

Here, **dynamic graph means dynamic constraint activation**, not learning a new graph topology at every diffusion step.

## Method at a glance

```text
schema + known rules + train statistics + optional offline proposals
                         |
          STAGE 1 — CONSTRUCT: verify / type / assign authority
                         |
                 typed constraint graph
                         |
 STAGE 2 — GUIDE: Applicable × Readable × Needed × Safe
                         |
 projection / numerical transport / categorical messages
                         |
                oversampled generated pool
                         |
             frozen distribution selector (C3)
                         |
 STAGE 3 — REPAIR: graph check + sparse quality-first swaps
                         |
                  final synthetic table
```

Stage 2 changes estimated values/logits during the reverse trajectory. Stage 3 never edits cells: it only changes which already-generated rows are retained. Repair becomes a no-op when the frozen C3 anchor is already graph-consistent.

## Novelty claims — explicit status

### Core claims with current implementation/evidence

- **Row × time × state active constraint graph:** factor applicability and intervention need are evaluated per row and reverse step, rather than using one fixed global penalty.
- **Typed actuation under a shared controller:** hard numerical factors route to projection, statistical factors to transport, and categorical factors to logit messages.
- **One graph at two decision scales:** the same endpoints, applicability, relation type, and authority control trajectory actions and final-set diagnostics.
- **Graph-deficit-gated proximal set repair:** start from a strong generic distribution selector, repair only factor-conditioned deficits, preserve generic quality, and abstain when no repair is justified.

### Boundary claims that must remain qualified

- Adult is the strongest complete case; Shoppers is coherent transfer evidence.
- Graph repair is not universally better than the frozen selector on every dataset.
- “Local SOTA” only means the same repository split/evaluator and rerun baselines.
- The work does not require claiming a new constraint-mining algorithm or a newly discovered topology at every step.

### Proposed extensions — novel but not yet verified

- **Counterfactual edge value:** activate an edge only when a cheap one-step lookahead predicts positive validity–fidelity gain.
- **Temporal edge lifecycle:** use onset, persistence, hysteresis, and survival traces instead of independent binary gates at each step.
- **Uncertainty-priced action budgets:** allocate intervention magnitude according to calibrated factor confidence and row-level preview uncertainty.
- **Graph residual witness:** carry unresolved trajectory deficits into Stage 3 as factor-specific repair priorities, directly coupling local failures to set-level swaps.
- **Trace-based explanation:** export per-row edge activation/action histories as an auditable explanation object.

These proposed extensions are research directions, not reported results. See `docs/NOVELTY_CLAIMS.md`.

## Where to start

1. Read `research_structure/README.md` for the presentation-aligned code map.
2. Read `docs/PROBLEM_METHOD_MAP.md` for the scientific structure.
3. Read `docs/CODE_WALKTHROUGH.md` for concrete implementation entry points.
4. Read `docs/NOVELTY_CLAIMS.md` for claim/evidence boundaries and optional new ideas.
5. Read `docs/EXPERIMENT_AND_EVIDENCE_MAP.md` for the job/result map.
6. Read `docs/PAPER_READY_RESEARCH_REPORT_20261022_VI.md` for the complete current report.
7. Read `docs/ABLATION_PLAN_10_GROUPS_20261022_VI.md` for the remaining experiment plan.
8. Read `docs/AGENT_HANDOFF.md` before asking another agent to edit or run the project.

## Important directories

```text
tabdiff/                                  upstream/model integration code
scripts/                                  upstream training and sampling helpers
analysis/llm_feedback_20260922/scripts/   research implementations and evaluators
analysis/llm_feedback_20260922/slurm/     exact experiment launchers
analysis/llm_feedback_20260922/evidence/  curated summaries, metadata, and logs
research_structure/                       presentation-aligned navigation façade
docs/                                     method, code, experiment, and handoff documentation
manifests/                                Git state, packaged-file hashes, omitted artifacts
environment/                              dependency snapshot
```

The chronological `analysis/` tree is deliberately retained for provenance. `research_structure/` is the clean conceptual façade; it points to real files without duplicating or silently rewriting historical code.

## Current evidence summary

- Adult final graph-deficit-gated repair: comprehensive fresh-seed result, `4/5` local baseline records.
- Shoppers graph-proximal repair: coherent transfer result, `3/5` local records.
- Frozen generic C3: strong independent confirmations on Magic, Diabetes, and Shoppers.
- Default/Magic/Diabetes graph-repair failures are retained as boundary evidence.
- News provides trajectory/timing/action evidence and a public-constraint case study.

“Local SOTA” means best among baselines rerun with the repository's local split/evaluator. It is not interchangeable with paper-reported SOTA from a different protocol.

## Data and checkpoints

Datasets, generated sample pools, checkpoints, and environments are deliberately excluded from Git. See `manifests/OMITTED_ARTIFACTS.csv` and provenance files under `analysis/llm_feedback_20260922/evidence/outputs/` for original server paths and hashes.

## Status

This snapshot is a handoff point, not a release. The method is sufficiently stable for paper writing, but final standardized multi-seed tables, runtime statistics, and several P0 ablations remain open. See `docs/CODEBASE_STATUS.md`.
