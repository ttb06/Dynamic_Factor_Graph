# Dynamic semantic constraints: narrowed research formulation

**Updated:** 2026-09-25  
**Evidence labels:** `[RUN]` executed here; `[PRELIM]` executed single-seed pilot; `[QUEUED]` submitted; `[INFERENCE]` interpretation.  
**Privacy:** API runs contain researcher-written contracts, schema metadata, and synthetic rows only. No real row is sent externally.

> **Newer synthesis:** `PROJECT_SYNTHESIS_20260926_VI.md` supersedes the job
> status and decision summary below. It includes the completed two-phase
> schedule, Default/Shoppers checkpoint audits, and target-scoped actuation
> experiment submitted on 2026-09-26.

## Decision

The most defensible project is not “LLM feedback at some diffusion steps” in general. It is:

> **Open-contract conditional diffusion:** compile a previously unseen natural-language semantic contract into a schema-grounded representation, condition a mixed-type diffusion model on it, activate it only where applicable, preserve exceptions, and measure transfer to unseen values, compositions, and tables against enforcement and repair baselines.

Intermediate-preview feedback remains an optional actuator/ablation. This gives a clean paper even if online LLM calls add no value.

The strongest trajectory-specific question emerging from the runs is a **reliability–actuation dilemma**: early previews are unreliable for subgroup applicability, but waiting for high-confidence previews may leave too few denoising steps to change the row. The current schedule experiments test whether global request conditioning early followed by row-selective gating late can resolve this trade-off. This is more precise than “ask an LLM at several timesteps.”

## What “dynamic” should mean

Separate three axes that are often conflated:

1. **Request-dynamic:** the contract `q` changes across generation requests and can be unseen during training.
2. **Row-dynamic:** applicability, exception state, and uncertainty `a(q,x)` vary across generated rows.
3. **Trajectory-dynamic:** an intermediate preview `x_hat_0(t)` provides new evidence and changes the active intervention during sampling.

SCTab-Diff, Harpoon, and CTDF already make the response at a reverse step depend on the current state. Therefore “state-dependent per-step guidance” is not a sufficient gap. The primary work should claim request- and row-dynamic semantics. Trajectory-dynamic feedback should be kept only if blank/shuffled-preview controls show unique value.

## New evidence 1: compile once beats repeated judgement under language shift `[RUN]`

The new bank deliberately falls outside the earlier template grammar:

- 220 synthetic cases, 10 contracts, 30 same-row/different-request pairs;
- held-out thresholds, reordered language, natural column aliases, conjunctive antecedents, `gt/neq/lte`, and rephrased exceptions;
- 70 satisfied, 70 violated, 60 not applicable, and 20 excepted cases;
- opaque API IDs; oracle DSL, labels, and pair IDs remain local.

| Model / mechanism | Four-way accuracy | Exact swap-pair accuracy | Exception false action | Tokens | Latency |
|---|---:|---:|---:|---:|---:|
| GPT-4.1-mini direct per-row judgement | .855 | .833 | .050 | 61,509 | 67.0 s |
| GPT-4.1-mini compile once + deterministic execute | **1.000** | **1.000** | **0** | **2,664** | **12.2 s** |
| GPT-5.4-mini direct per-row judgement | .636 | .333 | 0 | 61,755 | 41.7 s |
| GPT-5.4-mini compile once + deterministic execute | **1.000** | **1.000** | **0** | **2,658** | **7.2 s** |
| Oracle DSL | 1.000 | 1.000 | 0 | 0 | local |
| Old exact-template parser | unparsed | 0 | — | 0 | local |

Artifacts: `outputs/request_condition_aggregate_1033608/`.

Interpretation:

- The amortized compiler is about `23x` cheaper in tokens than direct judgement and is materially more reliable on request swaps.
- Failure of the old regex parser only establishes language shift beyond that parser; it does not prove general semantic understanding.
- Repeating an LLM call per row/step is now a weak default architecture. Compile once, validate, and execute locally is the stronger baseline.

## New evidence 2: finite-bank 100% hides schema-grounding errors `[RUN]`

Structural exact match is only `.50` for GPT-4.1-mini and `.70` for GPT-5.4-mini. Many differences are harmless ordering/redundancy, but both compilers turn canonical Adult values `Prof-school` and `Doctorate` into prompt surface forms `professional-school` and `doctorate` for one contract.

An automatically generated counterexample suite expands all antecedent category members and numerical boundaries:

| Compiler | Semantically exact contracts | Mean counterexample accuracy | Invalid categorical literals |
|---|---:|---:|---:|
| GPT-4.1-mini | 9/10 | .943 | 2 |
| GPT-5.4-mini | 9/10 | .943 | 2 |

Eight concrete synthetic counterexamples change the expected `satisfied/violated` state to `not_applicable`. This is a useful negative result: execution accuracy on a finite bank is not enough; compiled programs need schema-vocabulary validation and generated counterexamples before use.

Artifacts: `outputs/compiler_counterexample_audit_initial_20260925/`.

Grounded retries expose 104 allowed categorical values as schema metadata. Both models then reach 10/10 semantically exact contracts and zero invalid literals on the generated counterexample suite, for 3,215/3,209 tokens. This supports a concrete guardrail: vocabulary-ground, validate, and counterexample-test before a compiled program can control generation.

Artifacts: `outputs/compiler_counterexample_audit_1033618/` and `outputs/request_condition_aggregate_1033619/`.

## New evidence 3: semantic conditions transfer inside TabDiff `[RUN, three seeds]`

A small condition projection was inserted additively in TabDiff's latent MLP. The model starts from the existing Adult EMA checkpoint; only the condition projection and latent MLP core are fine-tuned. Training uses nine atomic contracts. Evaluation includes six unseen threshold values and three conjunctions never used during training. Each run uses 100 epochs / about 1,000 updates and 256 samples per contract.

The important comparator is gain over the model's own fine-tuned **null condition**, because fine-tuning alone changes unconditional prevalence.

| Representation / CFG | Seen atomic gain | Held-out threshold gain | Unseen composition gain | Composition cohort KS | Category coverage |
|---|---:|---:|---:|---:|---:|
| Request ID, 1x | +.102 | +.007 | +.023 | .146 | .975 |
| Request ID, 2x | +.168 | +.007 | +.023 | .146 | .975 |
| Typed contract, 1x | +.085 | **+.110** | **+.119** | .124 | .980 |
| Typed contract, 2x | +.150 | **+.171** | **+.194** | **.110** | .978 |
| Frozen text embedding, 1x | +.130 | +.104 | +.071 | .137 | .982 |
| Frozen text embedding, 2x | +.213 | +.169 | +.130 | .125 | .978 |
| Text + typed hybrid, 1x | +.131 | +.108 | +.088 | .135 | .977 |
| Text + typed hybrid, 2x | **+.217** | **+.175** | +.152 | .123 | .980 |

Request-ID vectors for unseen contracts are deliberately null, so their small held-out gain estimates drift/noise rather than semantic transfer. The typed representation passes the weak predeclared transfer gate (`>= +.05` on both held-out thresholds and compositions) and does so without observed duplicate collapse. Runtime was about 87–89 seconds on one T4 and peak allocation about 295 MB.

All three semantic representations pass the weak transfer gate in every seed, whereas request ID does not. Typed CFG-2 gives held-out-threshold gain `.171±.023` and unseen-composition gain `.194±.023`; request-ID gives `.007±.026` and `.023±.032`. Typed contracts remain strongest on unseen composition; hybrid is marginally strongest on held-out thresholds and seen atoms. Therefore this pilot does **not** support a claim that a foundation embedding is better than an executable AST. It supports conditioning on transferable semantics, with the LLM currently best justified as the compiler frontend.

This is still a mechanism pilot: contracts are simple numerical cohorts, `n=256` per contract/seed, and CTDF/Harpoon are not reproduced. The three-seed replication removes the most immediate variance concern.

Artifact: `outputs/condition_adapter_aggregate_1033615/` (plus per-representation run directories).

## New evidence 4: the actuator comparison is a Pareto problem `[RUN, three seeds]`

Three matched-checkpoint proxy jobs (`1033629–1033631`) compare unconditional sampling, deterministic final projection, and best-of-N selection. These are proxy mechanisms, not claimed reproductions of CTDF, Harpoon, or LCATS.

| Method | Denoiser passes/output | Held-out threshold satisfaction | Unseen composition satisfaction | Threshold cohort KS | Composition cohort KS |
|---|---:|---:|---:|---:|---:|
| Unconditional | 1 | .492 | .325 | .136 | .199 |
| Typed adapter, CFG 1 | about 1 | .687 | .532 | .099 | .124 |
| Best-of-2 | 2 | .689 | .498 | **.096** | .142 |
| Typed adapter, CFG 2 | about 2 | .748 | .608 | .106 | .110 |
| Best-of-4 | 4 | **.850** | **.712** | **.066** | **.093** |
| Final projection | 1 + local edit | **1.000** | **1.000** | .111 | .164 |

The matched three-seed result gives three useful reviewer-facing facts:

1. best-of-4 beats the current adapter on satisfaction/fidelity but costs four trajectories;
2. at roughly two denoiser evaluations, typed CFG-2 beats best-of-2 on satisfaction and composition KS;
3. final projection guarantees validity, but edits about 51% of threshold rows and 67% of composition rows and has much worse categorical TV (`.126/.185`) than typed CFG-2 (`.098/.095`), because clamping numeric targets does not generate the correlated cohort.

Thus the viable claim is a **budgeted Pareto region**, not maximum validity. On unseen compositions, typed CFG-2 strictly improves satisfaction, numeric KS, and categorical TV over best-of-2 at the same approximate denoiser budget; best-of-4 remains stronger at twice that budget. Artifact: `outputs/contract_pareto_20260925/`.

## New evidence 5: raw foundation embeddings are not numerically reliable `[RUN, paired-noise probe]`

The aggregate transfer scores above can hide whether the condition responds to the requested parameter. A causal probe therefore generates paired rows from identical sampling noise under low/high requests (`age 30→45`, `hours 30→45`, `education.num 10→13`) and then swaps, nulls, or averages the vectors.

| Representation | Correct directions | Mean high-minus-low target | Worst pair | Other-numeric change | Categorical change |
|---|---:|---:|---:|---:|---:|
| Typed AST | **3/3** | +.362 | **+.294** | .038 std units | .056 |
| Frozen text | 2/3 | +.937 | **−.648** | .113 | .136 |
| Text+AST hybrid | 2/3 | +.986 | **−.619** | .114 | .131 |
| Hybrid, AST-only ablation | **3/3** | +.096 | +.066 | .009 | .014 |
| Hybrid, text-only ablation | 2/3 | +.914 | −.659 | .109 | .138 |

Null and mean-vector controls produce exactly zero paired difference; swapping the typed vector reverses all three margins. Thus the typed condition is causally used, although weakly calibrated. Text and hybrid are stronger on average but invert the weekly-hours request: asking for 45 rather than 30 produces about `.62–.65` fewer hours. The full hybrid is dominated by its text branch, not protected by the typed prefix.

This rejects the raw version of H4. The more defensible foundation-model hypothesis is now **factorized grounding**: use a foundation representation for schema/semantic structure, keep operators and numerical/category parameters in typed channels, and add parameter-swap consistency.

Independent text-branch dropout (`1033735–1033738`) did not repair the failure. It strengthened the hybrid AST-only mean margin from `.096` to `.203`, but the full hybrid still inverted `hours 30→45` (`−.695`) and its text-only ablation was worse (`−.904`). This is a useful negative control: branch dropout does not make an ungrounded embedding numerically trustworthy. Typed parameters remain the safe default.

The denoiser-level paired monotonic loss (`1033763–1033768`) also failed as a guardrail. Its training loss fell to about `.015`, and three-seed aggregate transfer changed only slightly (CFG-2 threshold/composition gain `.178/.163` versus `.175/.152` without the loss), but the held-out `hours 30→45` terminal margin worsened to `−1.294`. Mean margin looks better only because the age pair jumps `+4.641`. Therefore a local ordering constraint on `x_hat_0` at random noise levels does not guarantee an ordered final sampling trajectory or interpolation between condition values. This narrows the next hypothesis to trajectory-level parameter consistency or explicitly typed actuation; simply adding a semantic regularizer is insufficient.

The endpoint control makes the failure precise. Without the loss, all three *seen* low/high pairs are correctly ordered, including `hours 20→50` at `+4.459`; the monotonic loss strengthens that endpoint to `+6.545`. Yet the unseen in-between `hours 30→45` pair remains inverted and worsens. The problem is parameter interpolation between requests, not total failure to use the condition or optimize the surrogate.

A cleaner factorization test lets the foundation embedding see only a parameter-free semantic template (which columns and relation family), while the typed AST is the sole carrier of operators and numerical thresholds. Within a contract family the text suffix is exactly identical across thresholds. In the paired-noise run it fixes all three held-out directions: age `+.166`, hours `+.171`, and education `+.264`; text-only produces exactly zero low/high difference, confirming that the parameter change can flow only through the typed channel. It also cuts other-numeric/categorical changes to `.023/.033`, versus `.114/.131` for the entangled hybrid.

Across three seeds at CFG 2, factorized threshold/composition gains are `.165±.025/.162±.034`. This is similar to the raw hybrid (`.175/.152`) and below typed AST on composition (`.171/.194`). Its mean paired margin (`+.200`) is also below typed AST (`+.362`). Therefore this is evidence for **safety by factorization**, not evidence that a foundation embedding adds value on Adult. The scientifically honest default remains typed AST; foundation semantics needs a held-out-schema/column test before it merits inclusion beyond the compiler frontend.

Artifacts: original swaps `outputs/condition_swap_aggregate_20260925_v2/`; monotonic aggregate/swap `outputs/condition_adapter_aggregate_1033768/` and `outputs/condition_swap_hybrid_mono_20260925_1033767/`; factorized aggregate/swap `outputs/condition_adapter_aggregate_1033796/` and `outputs/condition_swap_hybrid_factor_20260925_1033795/`.

## New evidence 6: row-wise applicability gating helps, but does not solve exceptions `[RUN, three seeds]`

Global conditional training can improve a rule either by changing its consequent or by making the antecedent subgroup disappear. A new implication suite therefore measures conditional violation, antecedent prevalence, exception prevalence, and drift among inactive rows. It contains category-dependent age/hours contracts, unseen thresholds/compositions, and an `unless occupation=Exec-managerial` exception. Three constructions use the same checkpoint, seeds, contracts, and training budget:

- `filter`: retain only rows satisfying the implication;
- `repair-preserve`: alter only the violating consequent and preserve antecedent/exception rows;
- `repair-gated`: the same repaired pairs, plus a two-pass sampler that derives a categorical preview and activates the condition only on applicable, non-exception rows.

The larger replication uses 2,048 samples per contract, three seeds, and CFG 1/2/4. At CFG 2, the row gate improves conditional violation over repair-preserve by `.032` on train, `.020` on unseen thresholds, and `.036` on extrapolation; composition changes only `.005`. Antecedent-rate error is unchanged or slightly better and inactive-target drift falls. The exception split is the boundary: violation becomes `.012` worse even though antecedent prevalence is better preserved. Categorical TV differs by at most `.002` at CFG 2.

Increasing the gated model to CFG 4 lowers violation further—by `.046` on thresholds, `.032` on compositions, and `.058` on extrapolation relative to repair-preserve—but introduces `+.028–.046` antecedent-rate error and higher TV on most splits. This is direct evidence for a validity–coverage/fidelity frontier, not a free enforcement gain. Absolute violation also remains high (`.530` threshold, `.408` composition, `.778` extrapolation at gated CFG 4). The mechanism evidence supports selective activation, while rejecting any claim of solved hard constraint enforcement or exception reasoning.

Artifacts: high-power `outputs/implication_adapter_aggregate_1033759/`; earlier `n=512` replication `outputs/implication_adapter_aggregate_1033748/`.

## New evidence 7: applicability previews have a measurable reliability window `[RUN, three seeds]`

The gated sampler was instrumented at every denoiser call. Its executable applicability decision from the unconditional categorical preview is compared with the applicability of the final generated row. This is a temporal-consistency diagnostic—not an independent semantic-accuracy label—but it directly tests whether an early dynamic gate is acting on stable evidence.

At CFG 2 across three seeds:

| Split | `t>0.75` accuracy / recall | `0.25<t<=0.50` accuracy / recall | `0.10<t<=0.25` accuracy / recall | `t<=0.10` accuracy / recall |
|---|---:|---:|---:|---:|
| Train | `.805 / .649` | `.935 / .961` | `.978 / .993` | `.996 / .998` |
| Unseen threshold | `.837 / .581` | `.954 / .985` | `.985 / .998` | `.998 / 1.000` |
| Unseen composition | `.962 / .131` | `.987 / .909` | `.996 / .972` | `.999 / .997` |
| Exception | `.671 / .980` | `.872 / .953` | `.950 / .979` | `.993 / .997` |

The early composition accuracy is misleading because the conjunction is rare: recall is only `.131`. Relevant-category agreement rises from roughly `.51–.60` at `t>0.75` to `.93–.98` by `t<=0.25`. Exception decisions converge latest. Thus an always-on per-row critic is poorly justified; the defensible schedule is to compile the contract before sampling but delay row-selective actuation until a validation-calibrated window (approximately `t<=0.25` here, and potentially `t<=0.10` for exceptions). Always-on versus delayed schedules are now being compared with paired noise.

The causal schedule comparison confirms a reliability–actuation dilemma. Versus never activating the condition, always-on gating reduces conditional violation by `.039` on train, `.029` on unseen thresholds, `.026` on compositions, and `.043` on extrapolation. Waiting until `t<=0.25` recovers essentially none of those gains (`−.001/.008/.007/.001` respectively), despite its much more reliable gate. `t<=0.10` is numerically almost identical to never acting. The exception contract is different: `t<=0.25` is best (`.668` violation versus `.678` always and `.690` never), but with higher antecedent-rate error. This rules out a simple “just call the critic late” solution. The new global-early→row-gated-late jobs test whether request-level conditioning can shape the trajectory before row-level evidence is trustworthy.

Artifacts: trajectory reliability `outputs/applicability_gate_trajectory_aggregate_1033778/`; delayed schedule `outputs/delayed_gate_schedule_aggregate_1033786/`.

## Formal hypotheses

### H1 — Schema-grounded compilation

A foundation model plus vocabulary/type validation and counterexample tests compiles unseen natural-language contracts more reliably than direct row judgement and a fixed template parser.

- Decisive split: held-out paraphrase, operator, categorical alias, and composition.
- Metrics: program execution equivalence on generated boundary cases, invalid literals, exception false action, cost.
- Kill: a generic deterministic parser matches the compiler on held-out language/tables.

### H2 — Contract-representation transfer

A typed or hybrid text+typed condition transfers to unseen thresholds and compatible compositions inside a diffusion generator; request IDs do not.

- Current signal: across three seeds, typed CFG-2 gives `+.171±.023 / +.194±.023` gain over null for held-out thresholds/compositions.
- Decisive next test: 3–5 seeds, more contract families, threshold extrapolation separated from interpolation, shuffled-condition control.
- Kill: transfer disappears after seed replication or is just unconditional drift.

### H3 — Selective applicability and exception preservation

Separating `scope`, `require`, and `unless` prevents antecedent evasion and rare valid-mode deletion better than treating every relation as a hard global rule.

- Benchmark strata: violated, satisfied, not applicable, exception.
- Current signal: in the 2,048-sample replication, executable preview gating lowers unseen-threshold/extrapolation violation by `.020/.036` at CFG 2 with no antecedent-prevalence cost, but gives almost no composition gain and is `.012` worse on the exception split. CFG 4 buys more validity by sacrificing antecedent fidelity.
- Baselines: hard global compiler, CTDF/Harpoon-style active loss, postrepair, rejection/best-of-N.
- Kill: oracle postrepair matches validity at lower distortion/cost and preserves exceptions equally well.

### H4 — Foundation semantics adds value beyond DSL parameters

Frozen foundation text representations or text+AST hybrids improve held-out language/table transfer over typed numeric parameters alone.

- Current state: raw text/hybrid is not supported. It transfers in aggregate but reverses the hours threshold direction; typed AST is correct on all three paired tests.
- Safeguard result: removing parameters from the text embedding restores all three directions and sharply reduces collateral changes, but three-seed transfer and paired magnitude do not beat typed AST. On Adult, the FM is justified as compiler, not yet as generator condition.
- Decisive split: column rename, unseen table, semantically equivalent paraphrase, adversarial number/operator changes.
- Kill: AST and a small parser equal or beat frozen embeddings; then retain LLM only as an interface/compiler, not as the generator condition.

### H5 — Online preview has unique information (optional)

Given a fixed contract representation, a calibrated preview verifier improves final semantic survival beyond `(contract,t)` schedules and final repair.

- Required controls: blank preview, shuffled preview, previous-clean/current-corrupt shortcut removal, matched denoiser calls.
- Current evidence: categorical applicability is not trustworthy early; relevant-category agreement becomes roughly `.93–.98` only by `t<=0.25`. But starting at `t<=0.25` loses almost all validity gain on ordinary contracts. A viable method must bridge this reliability–actuation gap (for example global-early/selective-late, lookahead, or rollback), not merely choose a late timestep.
- Kill: no unique gain; omit online LLM feedback from the main method.

## Method skeleton that is easy to defend

1. **Contract frontend:** NL request → typed AST (`scope`, `require`, `unless`, mutable fields, priority).
2. **Grounding guard:** validate columns/categories/types against schema; generate boundary/categorical counterexamples; reject or retry invalid programs.
3. **Condition encoder:** request ID, AST, frozen text embedding, and text+AST hybrid.
4. **Generator:** small adapter/FiLM in TabDiff with condition dropout and CFG.
5. **Selective verifier:** deterministic execution for executable clauses; calibrated FM verifier only for genuinely non-executable semantics.
6. **Two-phase actuation:** request-level/global condition while row state is uncertain, then applicability/exception-gated actuation after a validation-calibrated preview window; abstain or rollback on unstable gates.
7. **Actuation baselines:** learned condition, CTDF/Harpoon-style guidance, final repair, and best-of-N with matched budgets.

This method is intentionally incremental. The contribution is the open-contract interface, transfer benchmark, grounding/exception safeguards, and Pareto evidence—not a claim of inventing conditional diffusion.

## Three narrowed proposals

### P1 — Parameter-faithful open-contract diffusion (lowest risk)

Compile an unseen NL request into a schema-grounded typed contract and condition TabDiff through an adapter. Evaluate request swaps, unseen thresholds/compositions, and matched-budget best-of-N/repair. Unlike PromptLoop, the condition is an executable tabular contract rather than a revised prompt; unlike SCTab-Diff it is request-specific and tested on unseen parameters; unlike Harpoon/CTDF it learns a conditional cohort rather than only enforcing a fixed loss; unlike LCATS it is not solely post-generation repair. Current typed results support feasibility. The kill criterion is that typed conditioning cannot beat best-of-2 or a parser/repair stack on any fidelity-cost region. Estimated additional work: 2–3 weeks for multi-dataset adapters and full utility/privacy metrics.

### P2 — Reliability-calibrated two-phase actuation (best dynamic gap)

Use a global/request condition while the row preview is uncertain, then switch to a per-row active set once applicability is calibrated; unstable/exception cases abstain or rollback. The gap is not generic per-step guidance—Harpoon and CTDF already do that—but explicit measurement and control of applicability uncertainty over a mixed-type trajectory. PromptLoop/PPAD motivate intermediate feedback but do not solve contract applicability, exception preservation, or rare-mode coverage in tables. The decisive experiment is always-on versus delayed versus global-early→row-gated-late, with paired noise and matched active-pass budgets. Risk: late selection may be too weak, while early global conditioning may distort inactive rows. Estimated additional work: 3–5 weeks including rollback and cross-dataset replication.

### P3 — Exception-aware abstaining semantic verifier (higher risk)

Separate `applicable`, `satisfied`, `exception`, and `uncertain`; deterministic rules handle executable clauses, while a foundation verifier is invoked only for non-executable or ambiguous cases. Its action space is apply / abstain / rollback, not unconstrained free-form edits. This targets the observed exception failure and differs from LCATS/Zhao-style deterministic repair by making uncertainty and valid exception survival primary endpoints. The kill criterion is straightforward: if oracle rules plus postrepair match the frontier, omit the FM verifier. Estimated additional work: 4–6 weeks plus a genuinely semantic multi-domain benchmark.

## Immediate experiment queue

| Jobs | Experiment | Purpose |
|---|---|---|
| `1033604–1033608` | language/composition-shift direct vs compiler | completed evidence above |
| `1033609` | `text-embedding-3-large` contract vectors | completed; 18 strings, 512 dimensions, 208 tokens |
| `1033610` | adapter smoke test | completed |
| `1033611–1033615` | request-ID / typed / text / hybrid adapters + aggregate | completed |
| `1033616–1033619` | schema-grounded compiler retry, aggregate, counterexample audit | completed |
| `1033620–1033628` | two extra seeds for all four representations + three-seed aggregate | completed |
| `1033629–1033631` | matched unconditional / final projection / best-of-2/4 baselines, three seeds | completed; proxy baselines clearly labelled |
| `1033719–1033727` | smoke + three-seed filter-vs-repair-preserve conditional-implication adapters | completed |
| `1033729–1033731` | paired-noise correct/swapped/null/generic condition-vector probes | completed |
| `1033735–1033738` | hybrid with independent text-branch dropout + paired swap probe | completed; did not fix the hours inversion |
| `1033743–1033748` | two-pass row-applicability gate, three seeds + aggregate | completed; small consistent gain except on exception split |
| `1033750–1033759` | high-power implication matrix, 2,048 samples and CFG 1/2/4 | completed; confirms a validity–coverage frontier |
| `1033760–1033761` | Default and Shoppers TabDiff checkpoints | submitted; enables cross-dataset generation experiments |
| `1033800–1033801` | checkpoint fidelity + independent rule audit on Default/Shoppers | queued after successful cross-dataset training |
| `1033763–1033768` | paired monotonic-loss hybrid, three seeds + swap/aggregate | completed; local loss did not fix terminal hours inversion |
| `1033788–1033789` | train-endpoint swap probes with/without monotonic loss | completed; confirms interpolation failure despite stronger seen endpoints |
| `1033791–1033796` | parameter-free semantic embedding + typed parameters, three seeds/swap/aggregate | completed; fixes direction but does not beat typed AST |
| `1033774–1033778` | per-timestep applicability-gate reliability, three seeds | completed; supports delayed actuation |
| `1033782–1033786` | always-on vs delayed gate schedules with paired noise | completed; late reliable gates are mostly too late to act |
| `1033760–1033761`, `1033800–1033801` | Default/Shoppers checkpoints and audits | completed; good marginal fidelity and useful directional/exception rules |
| `1033803–1033807` | global-early then row-gated-late schedule, three seeds | completed; improves validity but increases inactive-row drift |
| `1040844–1040853` | target-scoped actuation, three guidance scales × three seeds | submitted; tests whether mutable-column masking removes collateral drift |

## Go/no-go

- **GO:** open-contract, schema-grounded compilation into typed parameters; the typed adapter transfers across three seeds and has a useful same-budget region versus best-of-2.
- **GO as the dynamic mechanism:** validation-calibrated row-selective actuation after the preview becomes reliable. Current evidence supports late gating for applicability, but not exception handling.
- **NO-GO:** repeated free-form LLM judgement at arbitrary reverse steps, raw text embeddings carrying numerical parameters, and local monotonic loss as a sufficient safeguard.
- **Foundation-condition result:** parameter-free text plus typed values fixes request direction, but does not outperform typed AST; a held-out-schema test is required before claiming value beyond compilation.
- **Pending decisive check:** whether delayed gating improves the validity/fidelity/cost frontier over always-on gating.

After those jobs, the highest-information step is cross-dataset replication on Shoppers/Default and an explicit exception/abstention mechanism. If factorized conditions do not beat typed AST, retain the LLM solely as a grounded compiler. If delayed gating does not match always-on validity with better fidelity or fewer active passes, omit trajectory feedback from the main method rather than forcing it.
