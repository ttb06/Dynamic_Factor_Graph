# Request-conditioned semantic constraint probe

Status: bank, local controls, and four API runs executed on 2026-09-24. This is a mechanism probe, not yet an end-to-end generation result.

## Hypothesis and gap

The useful foundation-model question may be broader than inference-time critique: can a model represent a *new generation request* as an executable, row-conditional semantic constraint, including applicability and exceptions? This differs from a fixed table prior because the correct label can change when the request changes while the row stays exactly the same. It also differs from Harpoon-style guidance because the initial request is natural language rather than an already executable differentiable loss.

The probe compares:

1. direct foundation-model judgement of `(request, synthetic row)`;
2. foundation-model compilation `natural language -> typed DSL`, followed by a deterministic local executor;
3. a deterministic parser for the controlled request templates;
4. an oracle DSL executor.

The second path is usable before training, as a conditioning representation during training, during sampling, or for post-generation selection/repair. It is therefore not tied to inference-time LLM calls.

## Executed bank

The bank contains 140 records derived only from synthetic Adult diffusion previews at five timesteps. No real row is used or sent externally.

- 30 paired tests use the exact same row under two threshold requests: age 24 under minimum-age requests 18 versus 30, and 35 weekly hours under minimum-hour requests 25 versus 45. A row-only plausibility prior cannot solve these pairs.
- Four-way labels are balanced enough to expose action errors: 50 satisfied, 50 violated, 20 not applicable, and 20 excepted.
- Explicit exception controls include `Married-AF-spouse` and `Never-worked + occupation='?'`.
- Explicit non-applicable controls have false antecedents.

Gold status is computed locally by an oracle DSL using the precedence `antecedent -> exception -> requirement`. Internal IDs, oracle DSL, gold labels, pair IDs and design labels are never included in API payloads. API case/request IDs are opaque SHA-256-derived tokens and cases are shuffled before batching.

## Local baseline result

Both the oracle executor and deterministic controlled-template parser achieve:

- four-way status accuracy: 1.00;
- exact request-swap pair accuracy: 1.00;
- execution equivalence: 1.00;
- exception false-action rate: 0.00.

This is an important negative control: an LLM is unnecessary when the request language is restricted to known templates. A foundation-model contribution must therefore be demonstrated under paraphrase, compositional, held-out constraint, or cross-table shift—not merely these six templates.

## API results

| Job | Model | Path | Four-way accuracy | Swap-pair accuracy | Exception false-action | Tokens |
|---|---|---|---:|---:|---:|---:|
| 1031780 | gpt-4.1-mini | direct request + row judgement | 0.864 | 0.867 | 0.000 | 37,544 |
| 1031781 | gpt-4.1-mini | NL request -> typed DSL -> local executor | 1.000 | 1.000 | 0.000 | 1,846 |
| 1031782 | gpt-5.4-mini | direct request + row judgement | 0.650 | 0.367 | 0.000 | 37,699 |
| 1031783 | gpt-5.4-mini | NL request -> typed DSL -> local executor | 1.000 | 1.000 | 0.000 | 1,853 |

Both compiler runs produced valid DSL for all six requests and execution equivalence of 1.00 on all 140 cases. Structural exact match was 5/6: for the `Never-worked` exception both models redundantly repeated `occupation='?'` inside the exception conjunction. Because the request antecedent already requires that predicate, the compiled program is execution-equivalent on the full bank. Direct gpt-5.4-mini additionally had a 0.25 false-action rate on not-applicable cases and 0.32 on satisfied cases; direct gpt-4.1-mini had 0.00 and 0.24 respectively.

The result supports **compile once, execute deterministically** over repeated direct FM judgement in this controlled setting: it is more accurate on request swaps, about 20x cheaper in tokens for this bank, and is not tied to inference-time calls. It does not yet show general language or table transfer.

The first aggregate job (1031784) failed only because pandas' optional `tabulate` package was absent; the renderer was replaced with a dependency-free implementation. Job 1031785 then exposed a Slurm comma-export issue and safely aggregated only one run. Final aggregate job 1031786 used a colon-separated ID list and loaded all four runs successfully. The four experiment jobs themselves completed successfully.

## Decision use and limitations

- If compile-then-execute preserves request swaps and exceptions better than direct judgement, a typed constraint representation is a stronger foundation-model direction than repeated free-form feedback.
- If both match the template parser, the next decisive experiment is language/constraint shift (held-out paraphrases and compositions), not more Adult sampling.
- If direct judgement is strong but DSL compilation fails, a learned/request-conditioned verifier or distillation target may be more feasible than symbolic compilation.
- This pilot has only six controlled requests on one schema. It does not establish unseen-domain generalization, natural-language breadth, sampling fidelity, rare-mode preservation, or benefit over post-repair.
- The synthetic rows are deliberately edited mechanism cases, so their class balance is not an estimate of natural generator error prevalence.

## Files

- Bank builder: `scripts/request_condition_bank.py`
- Deterministic DSL: `scripts/request_condition_common.py`
- API comparison: `scripts/request_condition_probe.py`
- Local baselines: `scripts/request_condition_local_baselines.py`
- Aggregator: `scripts/request_condition_aggregate.py`
- Slurm: `slurm/31_request_condition_api.sbatch`, `slurm/32_request_condition_aggregate.sbatch`
- Executed bank: `outputs/request_condition_bank_20260924/`
- Executed local controls: `outputs/request_condition_local_20260924/`
- Final four-run aggregate: `outputs/request_condition_aggregate_1031786/`
