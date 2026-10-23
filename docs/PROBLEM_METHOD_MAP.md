# Problem and method map

## 1. Scientific objective

Generate mixed-type synthetic tables that jointly preserve:

- univariate marginals;
- pairwise/joint trends;
- conditional and categorical dependencies;
- subgroup support/coverage;
- downstream utility;
- privacy/diversity;
- independently authoritative constraints where available.

Self-discovered relations are useful guidance signals, not automatic ground truth.

## 2. Three mismatches

- **Semantic mismatch:** a plausible relation may have exceptions or insufficient evidence.
- **Temporal mismatch:** a fixed penalty cannot know whether a row is readable, still changing, or already satisfied.
- **Set mismatch:** realistic rows can still form a final table with distorted factor-conditioned subgroup distributions.

## 3. One graph, three stages

### Stage 1 — Construct: factor discovery and verification

Sources remain separate: public/expert metadata, train-only statistical retrieval, and optional offline semantic proposals. Candidates are compiled into executable typed factors, checked for support/counterexamples/leakage, and assigned an authority role: enforce, score-only, observe, or reject.

### Stage 2 — Guide: dynamic trajectory guidance

At reverse step `t`, the clean preview determines which factors are applicable, readable, needed, and safe. Active factors route to typed projection, numerical transport, or categorical logit messages. Corrections are bounded and concentrated in middle steps.

### Stage 3 — Repair: graph-consistent output-set repair

An oversampled pool is first reduced by a frozen generic C3 selector. The same graph semantics audit the selected set. If graph deficit exceeds a train-only threshold, sparse membership swaps repair factor-state distributions subject to generic quality, coverage, and hard-factor constraints. Otherwise, the method abstains.

## 4. What is shared, and what is not

Shared across Stages 2 and 3:

- factor endpoints and type;
- applicability/support definition;
- authority and reliability;
- factor-conditioned states used for auditing.

Different across stages:

- Stage 2 changes trajectory values or logits;
- Stage 3 changes membership only;
- Stage 3 does not rerun diffusion or invent new values.

## 5. Core controls

- `B`: unmodified backbone.
- `G`: dynamic graph-guided generator.
- `R`: random/uniform subset of the generated pool.
- `old`: frozen generic C3 selector.
- `graph`: proposed deficit-gated graph-proximal repair.

## 6. Final locked repair configuration

- activation threshold: `0.001` graph relation loss;
- swap fraction: `0.25%`;
- minimum graph gain when active: `5%`;
- generic quality tolerance: `1.5%`;
- quality-first lexicographic selection;
- 256 train-only candidate draws.

## 7. Claim boundaries

Supported: comprehensive Adult result, coherent Shoppers transfer, strong frozen-C3 results, and mechanism evidence for dynamic/timed actions.

Not supported: universal cross-dataset SOTA, universal superiority over global guidance, or semantic/LLM proposals as the direct source of the main gain.
