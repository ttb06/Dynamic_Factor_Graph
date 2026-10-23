# Subgroup-conditioned dynamic constraint graph

## 1. Research problem

A synthetic-table generator may satisfy a relation globally while failing it in a specific operational regime, or a relation may be valid only inside one regime. A static constraint graph collapses these cases into one edge and one strength. This creates two symmetric errors:

1. **Over-enforcement:** a global edge is imposed on a subgroup containing legitimate exceptions.
2. **Under-enforcement:** a globally weak edge is ignored although it is reliable inside a meaningful subgroup.

The proposed setting is runtime semantic policy enforcement for mixed-type diffusion. The main difficulty is not that a subgroup is rare. It is that edge applicability, hardness and repair route depend on row context and may become readable only after the intermediate estimate `x_hat_0(x_t)` stabilizes.

Candidate application groups include workflow stage, calendar regime, customer/product configuration, acquisition channel, missingness pattern and schema version. Protected demographic groups remain audit-only and must not be used as hard validity constraints.

## 2. Proposed object

Let `G_global=(V,E)` be a typed prior graph compiled from schema, metadata and a runtime request. For row `i` at reverse step `t`, construct:

`G_i,t = Select(G_global, q_i, subgroup(x_hat_0_i,t), applicability_i,t, violation_i,t, uncertainty_i,t)`.

Each edge stores:

- typed antecedent and consequent;
- subgroup applicability or exception predicate;
- empirical hardness and support;
- allowed repair routes and actuated columns;
- provenance: user policy, documented invariant or empirical dependency;
- safety status: executable, soft-only, audit-only or rejected;
- uncertainty and abstention threshold.

This is more specific than changing graph topology using a heuristic similarity score. The graph changes only when a verified semantic state changes.

## 3. Role of the foundation model

The LLM should not repeatedly judge every row when deterministic execution is possible. Existing evidence favors compile-once use:

- direct record judgement required about 37.5k tokens and achieved accuracy `0.864`/`0.650` for GPT-4.1-mini/GPT-5.4-mini;
- natural language to typed DSL followed by deterministic execution achieved execution equivalence `1.0` with about 1.85k tokens.

The proposed LLM roles are therefore:

1. propose semantic subgroup predicates from schema, metadata and policy language;
2. distinguish plausible mechanisms from bookkeeping tautologies and accidental correlations;
3. label an edge as subgroup-active, subgroup-excepted, soft-only, audit-only or rejected;
4. propose multiple repair routes when a violation is ambiguous;
5. transfer this semantic selection across unseen tables.

Every proposal is type-checked and statistically verified. Raw real rows and heldout results are not sent to the API. Event-triggered LLM calls during sampling are reserved for non-executable ambiguity; they are not the default controller.

## 4. Verification pipeline

### 4.1 Train-only discovery

For an edge `A -> C` and subgroup predicate `G`, estimate violation rates inside `A & G` and `A & not G`. Require minimum support and control false discovery rate.

### 4.2 Interaction control

A reliability gap is insufficient because `G` may directly predict `C`. Measure edge uplift beyond each subgroup's consequent base rate:

`u_G = P(C | A,G) - P(C | G)`

`u_notG = P(C | A,notG) - P(C | notG)`.

The subgroup edge is retained only if `u_G-u_notG` is non-trivial and directionally consistent with the reliability gap. This rejects many proxy and tautological candidates.

### 4.3 Heldout replication

Subgroups and thresholds are selected only from train. Heldout data verifies support, direction, effect size and exception rate. Random split is an initial screen; temporal, site or source split is required for the main experiment.

### 4.4 Runtime activation

At selected diffusion steps, estimate:

`activation = readability × subgroup_probability × applicability × violation_probability × verified_edge_weight`.

Use hysteresis or multi-step agreement to prevent subgroup membership from flipping due to noisy previews. If membership or repair route is uncertain, defer or abstain. An independent final certifier handles residual violations and reports its edit burden.

## 5. Current evidence

### 5.1 Negative result: heuristic topology is insufficient

The existing dynamic-topology heuristic showed gains on some mixed/compositional conditions but was essentially tied with a fixed graph after the same certifier. Dynamic versus fixed after certification changed edit rate by `-0.0029`, KS by `-0.0003` and TV by `-0.0007`. This heuristic is stopped.

### 5.2 Four-state active set

Distinguishing `violated`, `satisfied`, `not applicable` and `valid exception` reduced applicability flips and distribution drift. Guidance plus certifier reduced repair edits by `42.6%` on Default and `71.2%` on Shoppers. Post-repair still had better KS/TV on simple one-target constraints, so this is evidence for selective activation, not superiority over repair.

### 5.3 Ambiguous repair

Job `1049914` created identical observed violations through two hidden corruption routes. Every repair baseline reached validity `1.0`, but fixed routes identified the correct edited column only `50%` of the time. A context router reached `73.45%`; oracle routing reached `100%`. Normalized pair error was `1.400` for consequent-only repair, `1.000` for the context router and `0.478` for oracle routing.

This demonstrates that violation rate cannot evaluate repair semantics and leaves a substantial attainable gap for better graph routing.

### 5.4 Preliminary subgroup audit

An initial seven-dataset audit found many heldout-stable reliability gaps. Example on Default:

- edge: `PAY_AMT4 > 0 -> PAY_AMT3 > 0`;
- subgroup: `PAY_4 == 0`;
- train violation: `2.3%` inside versus `26.6%` outside;
- heldout violation: `1.9%` inside versus `24.9%` outside.

The first audit also produced many redundant or tautological News edges. This is a useful failure case. The corrected audit adds subgroup-base-rate interaction control before any LLM selection.

Corrected job `1050323` retained and heldout-replicated:

| Dataset | Train candidates | Heldout replicated | Non-sensitive replicated |
| --- | ---: | ---: | ---: |
| Adult | 8 | 6 | 0 |
| Beijing | 3 | 3 | 3 |
| Default | 151 | 150 | 150 |
| Diabetes | 20 | 16 | 14 |
| News | 57 | 56 | 56 |
| Shoppers | 2 | 0 | 0 |

The signal is concentrated rather than universal: Shoppers loses all candidates after interaction and heldout controls, while Default remains strong. Adult candidates are sensitive and remain audit-only. Random splits and repeated variants of the same edge can still inflate apparent replication, so the next benchmark must cluster by original edge and add temporal/source shift.

### 5.5 LLM semantic selection

Array `1050324` evaluated at most 25 train-significant candidates per dataset. The API received schema and train aggregates only; heldout labels were joined locally.

Raw LLM selection was unsafe:

- GPT-4.1-mini: precision `0.846`, recall `0.985`, sensitive actuation `0.636`;
- GPT-5.4-mini: precision `0.881`, recall `0.776`, sensitive actuation `0.364`;
- conservative statistics: precision `0.955`, recall `0.627`, sensitive actuation `0`.

After a deterministic verifier blocks sensitive edges and actions whose direction contradicts the train interaction:

- GPT-4.1-mini: precision `0.930`, recall `0.985`, F1 `0.957`;
- GPT-5.4-mini: precision `0.941`, recall `0.716`, F1 `0.814`;
- conservative statistics: precision `0.955`, recall `0.627`, F1 `0.757`.

Compared only with the conservative threshold, cluster bootstrap gives GPT-4.1 verified:

- precision delta `-0.024`, 95% interval `[-0.109, 0.038]`;
- recall delta `+0.350`, interval `[0.132, 0.563]`;
- F1 delta `+0.199`, interval `[0.056, 0.374]`.

This comparison is not sufficient. The stronger matched baseline accepts every FDR-significant candidate after applying the same deterministic sensitive-edge block. It obtains precision `0.931`, recall `1.000`, F1 `0.964`, and therefore dominates GPT-4.1 verified (`0.930/0.985/0.957`). GPT-5.4 has a small precision increase but substantially lower recall and F1. Cluster bootstrap confirms both LLMs lose F1 against this verified-statistics baseline.

GPT-4.1 largely echoes the empirical direction; GPT-5.4 rejects more clinically or semantically dubious correlations but also loses replicated candidates. Both models activate protected demographic relations despite an explicit `audit_only` instruction. Self-reported confidence is not calibrated consistently and must not be used directly as graph weight.

Therefore **H4 is not supported in the current selection setting**. LLM selection adds no measurable value beyond train statistics plus deterministic safety. The dynamic subgroup graph can continue as a deterministic method, while an LLM contribution must be tested in a harder role unavailable to the statistical selector: proposing unseen natural-language subgroup semantics, resolving external-policy ambiguity or transferring a policy ontology to an unseen schema. The result does not establish semantic ground truth or cross-table superiority because the candidate payload is top-ranked and heldout splits are mostly random.

Job `1050327` measures when membership in the eight strongest replicated Default subgroups, and membership in the corresponding `antecedent AND subgroup` edge scopes, agrees with the final generated row across 30 reverse steps. It uses unconditional sampling, one seed and no LLM.

The aggregate trajectory is readable before the final step:

- at `t in (0.25,0.50]`, subgroup precision/recall is `0.958/0.974`, while full edge-scope precision/recall is `0.897/0.921`;
- at `t in (0.10,0.25]`, subgroup is `0.993/0.998` and edge scope is `0.981/0.992`;
- at `t > 0.75`, edge-scope recall is only `0.250`, so early activation is unsafe.

Two selected numeric exact-zero predicates had zero positive support in final diffusion samples although they have support in real data. Continuous diffusion almost never reconstructs the structural zero point mass exactly; their apparent accuracy `1.0` was trivial-negative accuracy. This exposes a representation requirement: numeric equality needs a tolerance or a zero-inflated/discrete-continuous column model.

Corrected job `1050329` excludes numeric exact-equality predicates; all eight candidates are evaluable. It confirms the temporal result:

- at `t in (0.25,0.50]`, subgroup precision/recall is `0.963/0.975` and edge scope is `0.906/0.930`;
- at `t in (0.10,0.25]`, subgroup is `0.995/0.997` and edge scope is `0.981/0.994`;
- at `t > 0.75`, edge-scope recall remains only `0.236`.

This supports a mid-to-late activation window with early abstention. It does not yet show that activation improves generation relative to group-aware post-repair.

### 5.6 Online subgroup projection: a saturated natural benchmark

Job `1050330` applied a CTDF-style numeric projection proxy to three repeated Default relations, with paired initial noise. It compared a global edge, preview-gated subgroup edge at `t<=0.5` and `t<=0.25`, an oracle frozen subgroup, subgroup-aware final repair, global final repair and a residual certifier.

The experiment is a scientifically useful negative result. Unconditional active-subgroup validity was already `98.99%`; all mid-denoising projection variants remained `98.99%`, while the certifier and final repair reached `100%`. The dynamic methods changed no valid exception rows. Global final repair changed `88.6%` of exception rows, mostly replacing a structural zero by `1`, but the normalized numeric displacement was tiny.

Consequences:

- near-hard natural relations are appropriate for auditing exception preservation, but they are too saturated to measure generation improvement;
- reporting only final validity would make a trivial repair look successful while hiding that it overwrote most valid exceptions;
- the next method screen must separate an `empirical_positive` regime from a controlled policy with a material threshold, and must report exception edits and conditional distribution fidelity;
- this result does not support superiority of online guidance over subgroup-aware final repair.

Jobs `1050333 -> 1050334` implement the next focused test. `1050333` trains one atomic structured-condition adapter for payment thresholds. It never sees row-wise subgroup composition. Dependent job `1050334` activates the atomic condition per row from the intermediate subgroup graph and compares static activation, dynamic preview activation, oracle frozen activation, subgroup repair and global repair. It evaluates both the empirical `>0` relations and a controlled `>=1000` policy regime. This tests whether a reusable condition layer can compose zero-shot with a dynamic graph, rather than training one generator per subgroup.

The first result falsifies the simplest form of S7. At `t<=0.5` or `t<=0.25`, dynamic activation is almost neutral relative to unconditional sampling. For the controlled request, mean satisfaction is approximately `0.81--0.82` versus `0.8195` unconditional; subgroup repair reaches `1.0`. The effect is not caused only by gating error: even the oracle frozen gate is ineffective when activated at `t<=0.5`.

Job `1050337` identifies a temporal alignment gap:

- oracle frozen scope, full-trajectory condition, CFG `2`: satisfaction `0.9407`, target KS `0.0896`, zero exception drift;
- dynamic preview scope, full trajectory, CFG `2`: satisfaction `0.8918`, target KS `0.1344`, but `48.4%` of valid exception rows change;
- late dynamic activation at `t<=0.5`, CFG `2`: satisfaction `0.8099`, target KS `0.2105`;
- unconditional: satisfaction `0.8195`, target KS `0.2084`;
- subgroup final repair: satisfaction `1.0`, target KS `0.1999`, zero exception drift.

Thus the condition adapter needs early influence, but the semantic edge scope is not reliable early. Applying noisy early gates improves validity but damages exceptions; waiting for reliable previews makes the condition ineffective. This is a stronger and more general gap than selecting a few LLM call timesteps. Job `1050338` adds the same residual certifier to early oracle and dynamic conditions to test whether the oracle route expands the final-validity frontier.

Job `1050338` confirms an informative upper bound for the controlled policy. At final validity `1.0`, frozen lookahead scope + full-trajectory condition + certifier has target KS `0.0896`, compared with `0.1999` for subgroup-aware final repair, with no exception-row target edits. Dynamic early gating + certifier has target KS `0.1344` but changes `48.4%` of exception rows. The frozen method still flips the antecedent on `6.35%` of rows, so it does not yet dominate on the full fidelity vector. On the already-saturated empirical-positive regime, conditioning is harmful: subgroup repair has target KS `0.1019`, whereas the frozen conditioned hybrid has `0.2382`.

The supposed oracle is deployable as a two-pass mechanism: first run an unconditional lookahead, compile a frozen active set from its final row states, then rollback to the same noise and resample. Job `1050339` tests this explicitly while anchoring the antecedent and subgroup columns to their first-pass trajectories. The intended contribution is not generic lookahead, but typed graph-based selection of immutable context versus resampled descendants, followed by independent certification.

### 5.8 Two-pass anchored graph and strong repair baselines

Job `1050339` confirms that trajectory anchoring removes both antecedent and subgroup flips and preserves every exception row. On the controlled-policy regime, anchored condition + certifier reaches validity `1.0` and target KS `0.1090`, versus `0.1999` for point-clamp subgroup repair. It remains harmful on already-saturated empirical-positive rules, so policy difficulty must be declared before evaluation.

This initial comparison used a weak repair baseline. Job `1050673` adds train-bank postrepair and evaluates fidelity against heldout rows. Empirical marginal repair applied only to violated rows reaches target KS `0.0699`, better than the all-applicable lookahead method, and therefore rejects any claim based only on beating clamp repair.

Job `1050675` then applies the four-state active set correctly: only `applicable + violated` rows receive the condition, while satisfied, non-applicable and valid-exception rows retain their unconditional trajectories. At equal validity `1.0`:

- four-state lookahead: target KS `0.1063`, cohort numeric KS `0.0684`, categorical TV `0.0416`, target-context correlation error `0.0527`;
- empirical marginal bad-only repair: `0.0699`, `0.0799`, `0.0301`, `0.0655`;
- point-clamp subgroup repair: `0.1999`, `0.0891`, `0.0301`, `0.1057`.

Thus empirical repair is better for the target marginal and categorical fidelity, while lookahead is better for whole-cohort numeric fidelity and dependency preservation. The methods form a frontier rather than a total ordering. Four-state lookahead changes the same rows as repair but usually uses much smaller target displacement than empirical resampling.

Jobs `1050676`, `1050680` and aggregate `1050685` confirm this over three sampling seeds and nine policy-seed pairs. Full-subgraph lookahead versus empirical repair has:

- active-drift delta `-0.151` SD, bootstrap interval `[-0.217,-0.086]`;
- target KS delta `+0.0349`, interval `[-0.0030,0.0673]`;
- cohort numeric KS delta `-0.0044`, interval `[-0.0112,0.0017]`;
- categorical TV delta `+0.0097`, interval `[0.0011,0.0177]`;
- correlation-error delta `-0.0151`, interval `[-0.0254,-0.0059]`.

Against clamp repair, lookahead significantly improves target KS, Wasserstein distance, cohort numeric KS and correlation error, at the cost of more displacement and categorical drift.

### 5.9 Intervention scope is a graph decision

Job `1050676` restricts conditional output to the target column. This preserves categorical fidelity but loses validity and dependency quality. Job `1050684` tests three scopes on the same policies:

| Scope | Target KS | Cohort numeric KS | Categorical TV | Correlation error | Active drift SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| target only | 0.1544 | 0.0822 | 0.0299 | 0.0944 | 0.0095 |
| BILL/PAY numeric subgraph | 0.1294 | 0.0705 | 0.0330 | 0.0609 | 0.0135 |
| full record | 0.1063 | 0.0684 | 0.0416 | 0.0527 | 0.0213 |

This monotone tradeoff supports a more precise method claim: the dynamic graph should select both active edges and the descendant actuation scope. Jobs `1050686 -> 1050687` are confirming the numeric-subgraph result on two additional sampling seeds.

The three-seed aggregate `1050687` confirms the ordering:

| Method | Active drift SD | Target KS | Cohort numeric KS | Categorical TV | Correlation error |
| --- | ---: | ---: | ---: | ---: | ---: |
| empirical bad-only repair | 0.1707 | 0.0733 | 0.0794 | 0.0333 | 0.0729 |
| target-only lookahead | 0.0098 | 0.1507 | 0.0825 | 0.0331 | 0.0885 |
| numeric-subgraph lookahead | 0.0136 | 0.1253 | 0.0748 | 0.0351 | 0.0595 |
| full-record lookahead | 0.0200 | 0.1082 | 0.0750 | 0.0430 | 0.0578 |
| clamp repair | 0.0063 | 0.1936 | 0.0880 | 0.0333 | 0.0974 |

Against empirical repair, numeric-subgraph lookahead uses `0.157` SD less displacement, while target KS is worse by `0.052`; categorical TV is statistically tied and numeric/correlation metrics favor lookahead but have intervals touching zero. Against clamp repair, it significantly improves target KS, Wasserstein, cohort numeric KS and correlation error. This is a stable non-dominated point, not overall superiority.

Job `1050689` extends the same mechanism to Shoppers using an existing independently trained adapter and three held-out categorical subgroup compositions. This is a controlled-policy cross-dataset pilot; it is not evidence that the earlier empirical Shoppers graph replicated, because that audit had zero retained subgroup edges.

The one-seed Shoppers pilot has sufficient active support (`140--424` rows per policy) and reproduces the frontier under a harder regime where unconditional satisfaction is only `0.368`:

- numeric-subgraph lookahead + certifier: target KS `0.271`, cohort numeric KS `0.161`, categorical TV `0.106`, correlation error `0.201`, active drift `0.867`;
- empirical bad-only repair: `0.199`, `0.176`, `0.120`, `0.228`, `1.117`;
- clamp repair: `0.646`, `0.221`, `0.120`, `0.177`, `0.277`.

The graph method again sacrifices target-marginal fit relative to empirical repair while improving whole-cohort numeric fidelity, categorical fidelity, dependency and intervention magnitude. Jobs `1050690 -> 1050691` are testing two more Shoppers sampling seeds.

### 5.7 Temporal-regime stress

Job `1050336` recombined prepared rows and ordered them using available time proxies. Candidate triples had already been discovered, so this is a robustness screen rather than a leakage-free temporal benchmark.

- Beijing: `3/3` candidates were evaluable and temporally stable.
- News: `39/57` were evaluable and `28/39` (`71.8%`) retained interaction direction and minimum late effect.
- Shoppers: neither candidate was evaluable under the month split.

Among the 56 News edges that replicated under the prepared random split, only `27/38` evaluable edges survived the time-regime criterion. Several title/keyword relations collapsed to zero interaction in the late regime. Random heldout replication is therefore optimistic for a non-trivial portion of the graph. A main-paper benchmark must perform candidate discovery from the early regime only and reserve the late regime for all threshold and effect evaluation. Until then, empirical edges require calibrated uncertainty and abstention rather than hard enforcement.

## 6. Hypotheses

### S1 — Stable conditional graph structure

After support, multiple-testing and interaction controls, some edge-strength differences replicate under heldout and source/temporal splits.

Falsification: effects disappear outside a random split or are explained by subgroup consequent prevalence.

### S2 — Subgroup graphs prevent over- and under-enforcement

An oracle subgroup graph improves the validity–fidelity–exception-preservation frontier over one global graph with matched actuation budget.

Falsification: global graph, calibrated soft edge or group-aware post-repair matches the oracle frontier.

### S3 — Subgroup state is readable during diffusion

Subgroup membership, active-edge set and repair-route ranking become stable at measurable timesteps. Activation after this point is more reliable than an arbitrary fixed schedule.

Falsification: membership remains unstable until the final step, in which case post-repair is sufficient.

### S4 — LLM adds semantic precision

LLM plus deterministic verification selects a higher fraction of heldout-replicated and semantically deployable subgroup edges than accepting all statistically significant edges, while retaining useful recall and producing zero sensitive-edge actuation.

Falsification: conservative statistics, a decision tree or a typed rule library matches the LLM.

Status: falsified for selection from the current statistical candidate bank. The matched verified-statistics baseline has higher F1 than both tested LLM selectors. Retain this hypothesis only for proposal of predicates not present in the bank, external policy interpretation or held-out-schema transfer.

### S5 — Trajectory-aware repair route beats final repair

For coupled constraints with multiple valid repair routes, subgroup/trajectory routing reduces wrong-column edits and counterfactual reconstruction error relative to deterministic final repair at matched final validity.

Falsification: density-aware or minimum-cost post-repair closes the oracle gap.

### S6 — Cross-table semantic transfer

A zero-shot foundation model can classify subgroup-edge roles on held-out tables using schema and train aggregates, followed by local calibration, more accurately than a table-specific lexical heuristic.

Falsification: performance depends on memorized dataset vocabulary or does not exceed a shallow classifier.

### S7 — Condition layer and graph compose without subgroup-specific training

An adapter trained on atomic global conditions can be activated row-wise by a preview-derived graph and improve the validity--fidelity--exception frontier for unseen subgroup compositions.

Falsification: the adapter changes unrelated rows, causes scope evasion, or is matched by subgroup-aware final repair. Job `1050334` is the first screen.

Status: the naive late-activation method is falsified. The full-trajectory oracle shows the condition layer itself can help, but the dynamic preview gate is not trustworthy early enough.

### S8 — Semantic-readability and controllability windows are misaligned

Standard condition adapters obtain most control from early reverse steps, whereas row-specific semantic scope becomes reliable only in middle-to-late steps. A lookahead/rollback controller or an adapter trained for late condition injection can close the oracle--dynamic gap without overwriting exceptions.

Falsification: an early dynamic gate matches the frozen oracle after calibration, or a late-activation-trained adapter still cannot improve over subgroup-aware repair.

Status: supported for the current adapter. Increasing late CFG to `4` or `8` in job `1050674` does not recover control and increases collateral drift, so the effect is not simply insufficient guidance strength.

### S9 — Actuation scope determines the fidelity frontier

A typed descendant subgraph provides a better intervention--fidelity tradeoff than either target-only actuation or full-record conditioning. The active graph must therefore encode allowed descendants, not only edge applicability.

Falsification: the numeric-subgraph ordering fails across sampling seeds or is dominated by data-assisted postrepair on all joint and marginal metrics.

## 7. Required baselines

### Constraint representation and selection

- global static graph;
- empirical subgroup graph without LLM;
- conservative statistical selector;
- decision-tree or RuleFit subgroup discovery;
- mixture-of-experts/gating network;
- LLM-only selector;
- verified LLM selector;
- oracle subgroup graph;
- shuffled subgroup and shuffled request controls.

### Enforcement

- no guidance;
- static prior/SCTab-Diff-style proxy;
- Harpoon-style differentiable rule guidance;
- CTDF/projection for executable constraints;
- global graph guidance;
- subgroup graph guidance;
- final-only projection/post-repair in LCATS/Zhao style;
- group-aware post-repair;
- best-of-N or branch selection;
- subgroup guidance plus residual certifier.

### Evaluation

- independent violation rate;
- exception and non-applicable preservation;
- subgroup-conditional marginal and joint fidelity;
- rare-mode coverage as a secondary stress test;
- wrong-column edit rate and repair-route recovery;
- downstream utility by subgroup;
- membership/edge flip rate across timesteps;
- runtime, GPU use, API calls and tokens.

## 8. Novelty boundary

Do not claim first per-step constraint enforcement, first LLM prior, first graph conditioning or first LLM repair. CTDF, Harpoon, SCTab-Diff, LCATS/Zhao and PromptLoop already cover substantial parts.

The defendable target is the combination of:

> verified subgroup-conditioned edge applicability and repair-route selection, activated from uncertain intermediate mixed-type states for unseen runtime policies, with independent certification and explicit exception preservation.

The subgroup component must outperform decision-tree/gating baselines. The diffusion component must outperform group-aware post-repair. The LLM component must outperform conservative statistical selection or be removed from the main method.

## 9. Paper viability

This direction is broader than rare conditional generation and aligns with data-management concerns: integrity constraints, heterogeneous data regimes, policy changes, explainable enforcement and data-quality guarantees. It can form an ICDE-style project if three conditions are met:

1. stable non-tautological subgroup effects on at least three datasets and one non-random split;
2. oracle subgroup graph expands the frontier over strong group-aware post-repair/projection;
3. the implemented selector recovers a meaningful portion of the oracle gain without damaging exceptions.

At present the project has a reproducible non-dominated frontier on two mixed-type datasets, not a SOTA claim. Matched-scope imputation confirms a dataset-dependent tradeoff, while best-of-8 constraint-aware imputation largely beats mid-denoising conditioning on Shoppers single-threshold policies. Therefore single-threshold results no longer support a primary conditioning claim. Sparse descendant scope expands the frontier but does not produce a universal winner.

The immediate falsification tests are now:

1. whether row-specific repair-route selection reduces counterfactual error relative to fixed-route diffusion imputation;
2. whether the gain survives independent certification and strong density-aware repair;
3. whether relational/non-local constraints reproduce it on at least three tables;
4. whether a foundation model improves route selection for semantic policies beyond an executable context router.

If test 1 fails, dynamic topology should be removed from the diffusion claim and retained only as a benchmark/analysis contribution. If tests 1–2 succeed, the method contribution can be framed as a verified solver-routing graph: four-state active-set inference, row-specific endpoint/descendant selection, matched-noise resampling and final certification. The LLM remains a compile-once interface for unseen natural-language policies; repeated row-level LLM judgement is not supported.
