# Research direction update: open-contract conditioning for tabular diffusion

**Audit date:** 2026-09-24  
**Evidence convention:** `[RUN]` executed here; `[SOURCE]` primary paper/code; `[INFERENCE]` interpretation; `[QUEUED]` submitted but not yet interpreted.  
**Privacy boundary:** API experiments use schema, fabricated constraints, and synthetic previews only. No real row is sent externally.

## 1. Decision first

The strongest direction is no longer “an LLM critiques every reverse step.” It is:

> **Open-contract conditioned tabular diffusion:** represent an external, request-specific semantic contract with a foundation model or typed program; learn or apply it selectively per row, preserve explicit exceptions, and test generalization to unseen constraints, compositions, and tables.

This need not call an LLM during sampling. A practical version compiles natural language once, embeds the contract once, and conditions a diffusion adapter or typed controller. Online trajectory feedback becomes an optional ablation, not the central claim.

Current go/no-go:

- **GO:** condition representation, request-conditioned applicability, exception safety, and unseen-condition/table generalization.
- **CONDITIONAL GO:** train a TabDiff condition adapter after the representation benchmark passes same-row/different-request and held-out-composition tests.
- **NO-GO as primary claim:** generic LLM row judge, repeated API calls at arbitrary timesteps, or “first constraint enforcement inside tabular diffusion.”

## 2. Important correction to earlier evidence

The focused API results in `DIRECTION_DECISION_20260924.md` (F1 around `.92`) are **not valid scientific evidence**. The API saw case IDs ending in `_clean` or `_corrupt`, clean/corrupt pairs were co-located, and generic controls were routed to candidate `none`. A suffix classifier could exploit the label. Those runs remain provenance/debug artifacts only.

The corrected benchmark below uses opaque IDs, randomizes case order before batching, retains private labels locally, and gives every case a real constraint. It is still a controlled `K=1` verifier upper bound, not a router or end-to-end generation result.

## 3. New executed evidence: condition representation matters

### 3.1 Benchmark

`dynamic_constraint_bank_20260924` contains 315 synthetic-only cases:

- 7 conditional constraints;
- 5 diffusion previews;
- balanced `violated / satisfied / not_applicable` states;
- 3 replicates per cell;
- explicit separation of antecedent applicability from consequent satisfaction.

The constraints cover spouse coding, age/education plausibility, and employment missingness. They are executable; therefore a deterministic evaluator is a 100% construction oracle and remains the required upper bound.

### 3.2 Corrected API results `[RUN]`

| Model / condition representation | Applicability acc. | Status acc. | Violation P | Violation R | Violation F1 | False action, N/A | False action, satisfied | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-4.1-mini, ID only | .822 | .686 | .685 | .600 | .640 | .114 | .162 | 81,696 |
| GPT-4.1-mini, natural language | .917 | .876 | .926 | .829 | .874 | .019 | .048 | 87,815 |
| GPT-4.1-mini, structured predicate shown to LLM | .911 | .784 | .789 | .676 | .728 | .010 | .200 | 91,721 |
| GPT-4.1-mini, NL + previous preview | .971 | .943 | .894 | .962 | .927 | .048 | .067 | 129,915 |
| GPT-5.4-mini, ID only | .702 | .546 | .542 | .495 | .517 | .371 | .048 | 82,062 |
| GPT-5.4-mini, natural language | .956 | .879 | .949 | .714 | .815 | .029 | .010 | 88,163 |
| GPT-5.4-mini, structured predicate shown to LLM | .933 | .851 | .911 | .686 | .783 | .038 | .029 | 92,071 |
| GPT-5.4-mini, NL + previous preview | .981 | .863 | .958 | .657 | .780 | .019 | .010 | 130,137 |

Artifact: `outputs/dynamic_constraint_reaggregate_20260924/`.

### 3.3 What this supports—and what it does not

Supported:

1. A relation identifier is not a sufficient semantic condition. Full natural-language meaning improves F1 by `.235` for GPT-4.1-mini and `.298` for GPT-5.4-mini over ID-only.
2. Passing a terse JSON predicate to an LLM is not equivalent to executing it. It underperforms natural language for both models. The correct architecture is therefore likely **NL → verified DSL → deterministic executor**, rather than asking an LLM to interpret a DSL row by row.
3. Missingness coupling remains the difficult family. Natural-language violation recall is only `.53/.47` for workclass missingness and `.67/.47` for occupation missingness (GPT-4.1-mini/GPT-5.4-mini), while common spouse/age rules are easier.
4. Larger/newer is not uniformly better: GPT-4.1-mini is stronger in violation recall here; GPT-5.4-mini is usually more conservative. Model-family effects need replication, not a scale claim.

Not supported:

1. The temporal result is not evidence that natural diffusion feedback is beneficial. The benchmark inserts a corruption into the current record while the previous preview is unmodified, so temporal difference is a shortcut. GPT-4.1-mini exploits it; GPT-5.4-mini does not. A true-feedback test must compare online preview against shuffled/blank preview and `(request,t)`-only controls on naturally evolving trajectories.
2. `K=1` supplies the target constraint. It tests execution, not retrieval/routing.
3. The rules are executable and construction-balanced. These scores do not show an LLM advantage over code.
4. Timestep status accuracy is non-monotonic and nearly flat after controlled edits. This bank should not select a denoising schedule.

## 4. Frozen foundation-representation probe `[RUN, diagnostic]`

Eight completed embedding jobs compare `text-embedding-3-small/large` with TF-IDF using leave-one-constraint-out linear probes. The most promising cell is the large NL condition+row embedding: three-way status macro-F1 `.810`, versus `.619` for TF-IDF and `.167` for condition-only.

This is only a feasibility signal. The current rows were constructed for a target rule, and the row-only serialization can reveal construction artifacts or the relevant column subset. It is not enough to claim cross-constraint reasoning. The next bank therefore uses the **same row under multiple requests**, full-row serialization, meaning-changing request pairs, and held-out condition templates.

Artifacts: `outputs/constraint_embedding_*_1031771` through `..._1031778`.

## 5. Same-row/different-request compiler probe `[RUN]`

A second leakage-controlled bank contains 140 synthetic cases, including 30 pairs where the **same row** is valid under one threshold request and invalid under another. It also balances explicit satisfied, violated, not-applicable, and exception cases. Gold programs and pair labels remain local.

| Model / mechanism | Four-way accuracy | Exact swap-pair accuracy | Action accuracy | Exception false action | Tokens | API latency |
|---|---:|---:|---:|---:|---:|---:|
| GPT-4.1-mini direct row judgement | .864 | .867 | .893 | 0 | 37,544 | 67.8 s |
| GPT-4.1-mini NL→DSL once + local executor | **1.000** | **1.000** | **1.000** | **0** | **1,846** | **11.5 s** |
| GPT-5.4-mini direct row judgement | .650 | .367 | .750 | 0 | 37,699 | 28.6 s |
| GPT-5.4-mini NL→DSL once + local executor | **1.000** | **1.000** | **1.000** | **0** | **1,853** | **4.6 s** |
| Oracle DSL / controlled template parser | **1.000** | **1.000** | **1.000** | **0** | 0 | local |

The two compilers structurally match 5/6 programs exactly. On the sixth they add a logically redundant predicate already guaranteed by the antecedent; execution equivalence over all 140 rows is still 100%.

This is the clearest mechanism result in the current round:

1. compiling once and executing deterministically dominates repeated direct FM judgement in accuracy, pair sensitivity, tokens (`~20×` fewer), and latency;
2. the perfect template parser means an LLM is unnecessary for this controlled grammar;
3. the next meaningful test is free paraphrase/unseen composition and cross-table transfer, not a larger direct judge;
4. this supports a foundation **condition interface** that is amortized or learned, not mandatory inference-time API feedback.

Artifact: `outputs/request_condition_reaggregate_20260924/`.

## 6. Novelty boundary after the 2026 literature audit

Use four distinct notions of conditioning:

- `S`: a static table-level specification/prior;
- `R`: a request or condition fixed for a trajectory but variable across generation requests;
- `T-response`: the score, gradient, or projection changes with `x_t,t` while the specification remains fixed;
- `T-condition`: the active rule/applicability itself is updated from the current state.

| Existing work | Already occupies | What remains |
|---|---|---|
| [CTSyn, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/778055b859a3731bbe0d92fa61655e80-Abstract-Conference.html) | Cross-table diffusion foundation model conditioned on schema metadata (`S`) | Constraint/request rather than schema conditioning; unseen contract/composition evaluation |
| [SCTab-Diff, preprint](https://arxiv.org/abs/2609.16069) | LLM-extracted, train-validated static prior inside every denoising step (`S + T-response`) | External request-specific contracts, applicability, exceptions, abstention |
| [Harpoon, ICLR 2026](https://openreview.net/pdf?id=G5g6tDg1ZE) | Train-once differentiable inference losses for arbitrary user constraints (`R + T-response`) | Non-executable semantics and verified selective activation; Harpoon remains a direct energy baseline |
| [CTDF, ICAIF 2025](https://arxiv.org/abs/2606.28674) | Training-free per-step Euclidean/KL feasibility maps for mixed-type TabDiff; hard functional rules, zero measured violations | No claim of first per-step mixed-type constraints; compare against CTDF and final projection |
| [SAGE, ACL 2026](https://aclanthology.org/2026.acl-long.174/) | Value-dependent sparse dependencies in autoregressive LLM generation (`T-condition`) | External semantic policy is distinct from MI-derived endogenous dependency |
| [PromptLoop, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_PromptLoop_Plug-and-Play_Prompt_Refinement_via_Latent_Feedback_for_Diffusion_Model_CVPR_2026_paper.html) | FM reads intermediate estimate and updates condition during diffusion | Tabular feedback alone is not novel; mixed-type contract/applicability/verification must carry the contribution |
| [LCATS, KDD 2026](https://github.com/SJTU-DMTai/LCATS) | LLM discovery/diagnosis plus optimized post-generation repair | Mandatory postrepair baseline |
| [Zhao et al., preprint](https://arxiv.org/abs/2608.15109) | Agent emits executable equations/inequalities/logical dependencies, validates with counterexamples, then repairs unchanged generators | Compiler+validation is not sufficient novelty; request shift and selective generation are needed |
| [EVS, ICML Structured-FM workshop 2026](https://openreview.net/pdf?id=1k9oK22A3R) | Noisy-LLM constraint extraction, train verification, utility-preserving selection | Same warning: extraction/filtering alone is occupied |

The defendable gap is the **intersection**, not any component alone:

1. an external contract changes across requests and may be unseen during training;
2. the model distinguishes violated, satisfied, not applicable, exception, and uncertain per row;
3. the conditioned generator generalizes across paraphrases, parameters, compositions, and ideally tables;
4. execution is independently verified and preserves legitimate exception/rare modes.

## 7. Primary proposal: Open-Contract TabDiff

### 7.1 Research question

Can a mixed-type diffusion generator trained over a distribution of semantic contracts generate from `p(x | schema, contract)` for unseen parameter values and compositions, without the cost and distortion of per-step projection, rejection, or post-repair?

### 7.2 Minimal method

1. Define a typed contract AST with `scope`, `require`, `unless`, `mutable_columns`, and priority/conflict semantics.
2. Generate train-supported contracts from each dataset using train only. Attach independently written NL paraphrases.
3. Encode the contract with three matched alternatives:
   - request ID/one-hot;
   - typed AST encoder;
   - frozen FM text embedding, optionally text+AST.
4. Add a small FiLM/cross-attention adapter to TabDiff's denoiser; use condition dropout for a CFG baseline.
5. Train with denoising plus request-swap contrastive loss: the same row/latent paired with a satisfied versus meaning-changing unsatisfied contract must not collapse to the same condition response.
6. Use the LLM only once to compile free-form requests; deterministic execution/verification owns correctness.

This is intentionally close to existing conditional diffusion. The contribution would be the open contract interface and its generalization benchmark, not a new diffusion equation.

### 7.3 Decisive hypotheses

**H-F1 — Contract generalization.** Text/AST conditioning beats request-ID conditioning on held-out threshold values and compatible unseen compositions, at matched seen-request fidelity.

- Pass: at least `+10 pp` all-constraints-satisfied on unseen compositions with marginal/joint fidelity within the baseline confidence interval.
- Kill: gain exists only for paraphrases of seen programs, not new values/compositions.

**H-F2 — Semantic representation contributes beyond row artifacts.** On same-row/different-request pairs, the method gets both labels/actions correct and degrades under shuffled contracts.

- Pass: request-swap pair accuracy `>= .85`; shuffled-contract performance near chance; candidate Recall@2 `>= .95` at `K=16`.
- Kill: row-only or template parser ties the FM on held-out templates.

**H-F3 — Learned conditioning has a Pareto region.** It beats CTDF/Harpoon-style guidance, rejection, and final repair for at least one nontrivial low-support or compositional request under matched denoiser/verifier budgets.

- Pass: same or better satisfaction, lower non-mutable distortion, and better rare/exception coverage or runtime.
- Kill: oracle final repair or CTDF Pareto-dominates it.

**H-F4 — Cross-table semantic transfer.** A schema+contract encoder improves unseen-table adaptation over schema-only CTSyn/TabDiff-style conditioning.

- Pass: leave-one-dataset-out improvement survives column-name paraphrases and program-equivalent renaming.
- Kill: results are driven by overlapping column names or one dataset.

**H-F5 — Online trajectory evidence is optional and testable.** After fixing the contract, current preview adds value beyond `(contract,t)` and a precomputed schedule.

- Pass: online preview improves natural violation/action survival with shuffled-preview and blank-preview controls.
- Kill: no gain; keep the trained condition model and remove the feedback loop.

## 8. Secondary proposal: condition foundation verifier

Train or distill a cross-table verifier over `(schema, contract, row)` with explicit applicability/exception/abstention heads. This can be useful independently of where intervention occurs and is cheaper than a full cross-table generator.

Strong evaluation requires:

- held-out relation templates, datasets, parameters, and compositions;
- same-row/different-request pairs;
- `K={1,4,8,16}` hard distractors and none-of-the-above;
- naturally occurring held-out exceptions as well as controlled counterfactuals;
- deterministic parser/oracle DSL, direct FM judgement, and shuffled-contract controls.

This is a viable paper only if cross-contract/table transfer is real. A single-dataset Adult verifier is not enough.

## 9. Cross-dataset audit `[RUN]` and expansion

A train-only discovery/held-out audit completed on all three local datasets (`job 1031779`):

| Category | Candidates |
|---|---:|
| empirical executable baseline | 80 |
| near-hard dynamic benchmark | 95 |
| probabilistic, not hard validity | 334 |
| sensitive/demographic, audit only | 85 |
| target-related distributional association | 57 |
| insufficient support | 132 |
| subgroup/shift stress test | 5 |

The main empirical lesson is directionality and hardness, not the raw count of mined rules:

- Shoppers `Administrative=0 → Administrative_Duration=0` has 0 observed exceptions in both splits, while the reverse has `2.20%` train and `3.14%` test exceptions.
- `Informational=0 → Informational_Duration=0` is also exact empirically, while the reverse has about `2.3%` exceptions.
- `ProductRelated_Duration=0 → ProductRelated=0` is false for more than 90% of applicable rows, despite lexical similarity to the valid one-way form.
- Adult `Own-child → age<30` has about `19.7%` exceptions in both splits and is explicitly unsafe to enforce.
- Default manual temporal-looking rules are mostly probabilistic: examples show roughly `5%` and `28–33%` held-out/train violations.

This supplies a concrete benchmark rationale: a useful condition model must represent direction, applicability, hardness, and exceptions. Only 80/788 candidates are suitable executable baselines; “plausible dependency” must not be equated with validity.

Artifacts: `outputs/cross_dataset_constraint_20260924/`.

### Dataset expansion

Immediate local datasets:

- Adult: mixed type, current trajectory checkpoint, but demographic rules must be framed as legacy dataset coding rather than real-world norms.
- Shoppers: count-duration implications, rate ordering, revenue/value conditional relations.
- Default: coded categorical ranges and temporal bill/payment relations.

Useful extensions appearing in adjacent FM/LLM tabular work:

- SAGE suite: Adult, HELOC, Iris, California Housing, CDC Diabetes, Myocardial Infarction Complications;
- EVS-style structural datasets: HELOC, FAULTS, WIDS;
- CTDF-style finance: Airbnb/Lending Club if storage/licensing permits.

Iris is too simple as a main benchmark but useful for smoke tests. California is numeric-only. Adult/Shoppers/Default plus HELOC or MIC provide a better first mixed-type set. Discover rules only on train; freeze templates before looking at test exceptions.

Required splits:

1. seen template, unseen paraphrase;
2. seen operator, unseen threshold/value;
3. unseen compatible composition;
4. held-out exception family;
5. held-out dataset/table;
6. column rename/paraphrase stress test.

## 10. Baselines that must appear

- unconditional TabDiff;
- TabDiff feature/class conditional or imputation CFG where applicable;
- request-ID, AST, frozen text embedding, and shuffled-condition adapters;
- static prior/cross-attention proxy for SCTab-Diff;
- Harpoon-style differentiable loss;
- CTDF-style per-step mixed-type feasibility projection;
- final-only projection/postrepair, including LCATS/Zhao-style label as a proxy unless reproduced;
- best-of-N/rejection with matched denoiser passes;
- deterministic NL template parser and oracle DSL executor;
- direct API judge and, only later, distilled verifier.

## 11. Metrics and reviewer-facing safeguards

Report request satisfaction together with:

- four-way applicability macro-F1 plus exception and inactive false-action rates;
- request-swap pair accuracy and candidate Recall@B/MRR;
- antecedent evasion and immutable-guard retention;
- non-mutable-column distortion to the matched-noise baseline;
- conditional marginal/joint fidelity, subgroup fidelity, rare-mode coverage, and diversity;
- downstream utility as secondary evidence;
- calls, tokens, verifier evaluations, denoiser passes, runtime, and GPU memory;
- bootstrap intervals over seeds and request templates, not individual duplicated rows.

Never use the guiding rule/LLM as the only evaluator. Never call a controlled-corruption result a natural trajectory result. Never claim FM necessity if an executable parser ties it.

## 12. Current jobs and next decision

| IDs | Purpose | Status at report update |
|---|---|---|
| `1031754–1031761` | first dynamic batch | cancelled after detecting label-bearing IDs |
| `1031762–1031769` | corrected opaque-ID condition-representation sweep | completed |
| `1031770` | scheduled aggregate | completed but Slurm export preserved only first comma-separated ID; superseded by manual correct aggregate |
| `1031771–1031778` | frozen embedding vs TF-IDF, small/large, four input representations | completed; diagnostic only |
| `1031779` | train-only cross-dataset constraint and exception audit | completed |
| `1031780–1031783` | same-row/different-request direct vs compile-execute | completed |
| `1031784–1031785` | scheduled aggregate retries | partial (missing optional package, then export-list truncation); superseded by correct manual aggregate |

The highest-value next implementation after these jobs is a **small oracle-contract adapter on TabDiff**, not another online LLM sweep. Use 3–5 request families, held-out thresholds/compositions, and compare one-hot versus AST versus frozen text embeddings. Only add trajectory feedback if the online-preview ablation later shows unique information.

## 13. 2026-09-25 continuation

The proposed next implementation has now been executed at pilot scale. See `DYNAMIC_SEMANTIC_CONSTRAINTS_20260925.md` for the current formulation and exact artifacts.

Key updates:

- A harder 220-case language/composition-shift bank again shows compile-once + deterministic execute at 100%, while direct row judgement reaches `.855` (GPT-4.1-mini) and `.636` (GPT-5.4-mini), using about `23x` more tokens.
- Counterexample expansion invalidates the unqualified 100% compiler claim: both compilers are semantically exact on 9/10 contracts and emit two out-of-vocabulary category literals on the remaining request. Schema-vocabulary grounding retries are submitted.
- Three TabDiff condition-adapter seeds completed. Against each fine-tuned model's null condition, typed contracts improve held-out-threshold satisfaction by `.110/.171` and unseen-composition satisfaction by `.119/.194` at CFG 1/2. Request-ID transfer controls give only `.007/.023`.
- Frozen text and hybrid conditions also transfer, but neither beats the typed AST on unseen composition. A three-seed Pareto audit shows typed CFG-2 dominates best-of-2 on unseen compositions at similar denoiser budget; best-of-4 is stronger at twice the budget, while final projection gets 100% validity with substantially worse correlated categorical fidelity.
