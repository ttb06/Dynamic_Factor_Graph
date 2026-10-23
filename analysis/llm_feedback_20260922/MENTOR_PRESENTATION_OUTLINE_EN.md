# Mentor Presentation Outline

## Working title

**LLM-Guided Feedback During Tabular Diffusion: From Open-Ended Critique to Verified Typed Transactions**

Suggested duration: 30–35 minutes plus discussion. The main deck is 25 slides; detailed paper cards and extra result tables go to the appendix.

---

## Part I — Motivation and research landscape

### Slide 1 — Title

**LLM-Guided Feedback During Tabular Diffusion**  
*Can record-specific semantic feedback improve validity without sacrificing fidelity, rare modes, and cost?*

Include:

- Your name, group, date.
- Small subtitle: “Exploratory results and hypotheses for refinement.”

Opening sentence: this is not a claim that LLM feedback inside diffusion is new; the goal is to identify a tabular-specific contribution that survives strong rule-guidance and post-repair baselines.

### Slide 2 — Why distributional fidelity is not enough

Show one mixed-type synthetic row with:

- plausible marginals;
- an education/education-level mismatch;
- inconsistent spouse/sex/marital fields;
- a rare but valid row that should not be repaired.

Message:

> A generator can match global statistics while producing locally inconsistent records; however, aggressively enforcing plausible rules can delete legitimate exceptions.

Frame three objectives:

1. semantic validity;
2. distributional fidelity and utility;
3. exception and rare-mode preservation.

### Slide 3 — The design space: when and how knowledge enters generation

Use a 3-column timeline:

| Before sampling | During denoising | After generation |
|---|---|---|
| Static metadata prior | Differentiable constraint | Detection and repair |
| Representation selection | Black-box reward | MILP/rule correction |
| LLM-discovered rules | Intermediate-preview feedback | Regeneration |

Add a second visual axis:

- executable rule;
- learned differentiable reward;
- LLM semantic judgment;
- external tool/ontology.

Place papers on the map: LLM-TabLogic and SCTab-Diff on the left; Harpoon, CDD, PromptLoop, PPAD, Free²Guide and TTGen in the middle; LCATS and Zhao et al. on the right.

### Slide 4 — Diffusion backbones for mixed-type tables

Present two backbone options:

**TabSyn — ICLR 2024**

- Transformer VAE maps numeric and categorical columns into a continuous latent.
- Score-based diffusion operates in latent space.
- Advantage: unified continuous representation.
- Limitation for this project: feedback must be mapped back through the decoder/latent space; VAE reconstruction adds uncertainty.

**TabDiff — ICLR 2025**

- Direct joint diffusion over numerical and categorical features.
- Feature-specific noise schedules.
- Easier to decode an intermediate clean estimate and apply typed column actions.

Bottom-line banner: **TabDiff is the current experimental backbone because intermediate mixed-type actions are easier to inspect and control.**

### Slide 5 — Constraint guidance without LLMs

**Harpoon — ICLR 2026**

- Applies differentiable loss guidance during reverse diffusion.
- Strong baseline for imputation, inequalities, and geometric constraints.
- Establishes that “constraints inside denoising” is not itself novel.

**Constrained Discrete Diffusion — NeurIPS 2025**

- Projects categorical distributions under constraints during discrete denoising.
- Relevant to categorical remasking/projection.

**GRDM — NeurIPS 2025**

- Joint graph diffusion for relational databases and foreign-key structure.
- Relevant future backbone for multi-table semantic relations, not the present single-table pilot.

Research implication: any LLM method must justify why the signal cannot be expressed as a cheap differentiable or executable constraint.

### Slide 6 — LLM knowledge before sampling

**LLM-TabLogic — preprint**

- LLM discovers hierarchical, mathematical, and temporal relations from metadata.
- Relations determine compression and reconstruction around latent diffusion.
- LLM changes representation before sampling; it does not inspect intermediate samples.

**SCTab-Diff — 2026 preprint**

- LLM extracts and validates semantic priors from metadata.
- Static priors condition the score network throughout reverse denoising.
- Current-row context is used, but the prior itself is not updated by per-record critique.

Gap left open: **record- and timestep-specific decisions**, not simply “LLM prior inside denoising.”

### Slide 7 — Constraint discovery and post-generation repair

**LCATS — KDD 2026**

- LLM discovers executable local constraints and validates them on training data.
- TabSyn generates data.
- Diagnostics, repair weights, and per-row optimization repair final samples.
- Strongest conceptual baseline for whether mid-sampling intervention is needed.

**Zhao et al. — 2026 preprint**

- Agent proposes equations, inequalities, and logical dependencies.
- Counterexamples refine the rules.
- Generator remains unchanged; correction is post-processing.

Question to emphasize:

> If repair at `t=0` gives the same validity–fidelity frontier at lower cost, why intervene earlier?

### Slide 8 — LLMs as generators or joint components

**SAGE — ACL 2026**

- LLM is the primary generator.
- Value-aware pseudo-features create record-dependent dependency graphs.
- Shows that dependencies may change with values/subgroups.
- Limitation: pairwise graph can miss higher-order relations.

**AFT-Tab — ACL 2026**

- CTGAN generates structured columns; an LLM generates long text.
- A discriminator provides training-time feedback to both components.
- Relevant only when text–structured consistency is central.

**TabDLM — 2026 preprint**

- Continuous diffusion for numeric columns plus masked language diffusion for categorical/text columns.
- Joint bidirectional denoising, but expensive sampling and schedule coupling remain concerns.

Decision: defer the long-text branch until a suitable dataset/backbone is selected.

### Slide 9 — Original feedback-loop ideas from image diffusion

**PromptLoop — CVPR 2026**

- A multimodal LLM observes a denoised estimate at selected steps.
- It revises the prompt for the remaining trajectory.
- Therefore, “LLM observes an intermediate diffusion sample and updates conditioning” is already established.

**PPAD — 2025 preprint**

- Uses lookahead previews, error diagnosis, rollback and resampling.
- Important negative lesson: feedback from noisy previews can degrade quality.

Tabular opportunity: previews are structured, actions can be typed by column, and validity/exception coverage can be measured with independent verifiers.

### Slide 10 — Black-box reward and branch-selection ideas

**Free²Guide — ICCV 2025**

- Uses a black-box vision-language reward during diffusion without differentiating through the evaluator.
- Motivates branch selection/path-integral guidance.

**TTGen — CVPR 2025 workshop**

- LVLM revises prompts from clean-latent estimates.
- A separate score selects branches.
- Workshop paper, not a main-conference result.

Baselines implied for this project:

- best-of-N with a verifier;
- sparse black-box scoring;
- rollback/resampling;
- post-repair with the same action compiler.

### Slide 11 — Novelty boundary after related work

Use a “cannot claim / may claim” split.

**Cannot claim**

- LLM + diffusion;
- semantic prior conditioning;
- constraints during denoising;
- intermediate-preview feedback;
- generic adaptive scheduling.

**Potential contribution**

- record-specific relation routing for mixed-type rows;
- calibrated abstention for exceptions and rare modes;
- typed numeric/categorical actions with transaction and rollback semantics;
- evidence-based selection of when to call an expensive critic;
- direct comparison against the same compiler used as post-repair.

---

## Part II — Proposed mechanism and research questions

### Slide 12 — Initial idea and the refined architecture

Show two pipelines.

**Initial idea**

`x_t → decode x̂₀ → open-ended LLM critique → edit → continue denoising`

**Refined architecture**

`x_t → readable x̂₀ → relation router → focused LLM verifier → executable compiler → typed action → independent check/rollback → continue`

The LLM returns only:

- candidate status: `violated / satisfied / not_applicable / abstain`;
- relation ID;
- involved columns;
- confidence and evidence code.

The LLM never directly writes arbitrary values into the sampler.

### Slide 13 — Core research questions

Present five questions, not a long hypothesis list:

1. **Readability:** when is `x̂₀(x_t)` stable enough to judge?
2. **Need:** which errors cannot be handled by static rules or priors?
3. **Actuation:** which typed action survives the remaining denoising steps?
4. **Comparison:** does mid-sampling feedback beat post-repair or best-of-N?
5. **Reliability/cost:** can routing and abstention make LLM calls trustworthy and sparse?

Optional sixth question for future work: bidirectional text–structured correction.

### Slide 14 — Experimental protocol and evidence labels

Clearly distinguish:

- **Natural generation:** full 8,000-epoch Adult TabDiff checkpoint.
- **Controlled stress test:** deliberately injected semantic inconsistencies.
- **Proxy baseline:** mechanism implemented in the current backbone, not a paper reproduction.
- **Executed vs pending:** only completed outputs support claims.

Datasets currently profiled:

- Adult: 6 numerical + 9 categorical columns;
- Default: 14 numerical + 10 categorical;
- Shoppers: 10 numerical + 8 categorical;
- none contains long text.

Primary experiments currently use Adult because the checkpoint and interpretable relations are available.

---

## Part III — Empirical findings

### Slide 15 — H1: when is the intermediate preview readable?

Plot numerical MAE and categorical accuracy against timestep.

Key table:

| `t` | Numeric MAE | Categorical accuracy | Decoded hard-rule violation |
|---:|---:|---:|---:|
| .03 | .0040 | 99.15% | 1.48% |
| .10 | .0114 | 97.37% | 3.30% |
| .25 | .0760 | 92.77% | 13.19% |
| .40 | .2724 | 87.46% | 23.78% |
| .70 | 1.168 | 72.95% | 83.85% |

Conclusion:

- A defensible critic window begins around `t≤.10` for this checkpoint.
- Numeric and categorical features stabilize differently.
- Calling an LLM at arbitrary timesteps is not justified.

### Slide 16 — Rule audit: hard rules, near-hard rules, and exceptions

Show three example classes:

**Hard/executable**

- `education → education.num`: zero train/test violations.
- nonnegative gain/loss and positive weekly hours.

**Near-hard with empirical exceptions**

- spouse/sex/marital relations: approximately `0.008–0.19%` violations.

**Unsafe rule**

- `Own-child → age < 30` has approximately `19.7%` valid counterexamples.

Takeaway: semantic plausibility is not the same as an invariant. Rare-mode preservation must be a first-class metric.

### Slide 17 — Strong backbone changes the problem

Use the full-checkpoint numbers:

| Method | Hard violation rate |
|---|---:|
| No guidance | **1.76%** |
| One late typed correction | 0.20% |
| Targeted recheck | 0.13% |
| Fixed rule every step | 0% |
| Post-repair | 0% |

Message:

- The full TabDiff model already learns Adult hard rules well.
- Hard-rule enforcement is not a convincing use case for LLM feedback.
- Evaluation must move to exception-bearing or non-local semantic relations.

### Slide 18 — Typed actuation matters more than repeated critique

Show the full-checkpoint actuator result:

| Action | Final hard violation |
|---|---:|
| No guidance | 1.56% |
| `x̂₀` correction at `.03` | 0.26% |
| State projection at `.03` | 0.13% |
| State projection at `.10` | **0.065%** |
| State recheck `.10→.03` | **0.065%** |
| Post-repair | 0% |

Interpretation:

- Numeric changes made only through predicted `x̂₀` can be attenuated by later denoising.
- Direct typed state projection is much more durable.
- Rechecking adds little once the actuator is correct.
- Mid-sampling still has no demonstrated Pareto advantage over repair.

### Slide 19 — Controlled semantic stress test

Describe the bank:

- 220 synthetic cases;
- five timesteps: approximately `.414, .241, .172, .103, .034`;
- five corruption types;
- paired clean controls;
- rare-valid country and exception-valid `Own-child, age=45` controls;
- no real rows sent to external APIs.

End-to-end mechanism result:

- corruption without feedback: intended violation remains `100%`;
- oracle rollback: `0%` intended violation;
- static verifier rollback: `0%` intended violation;
- categorical agreement to clean after rollback: approximately `.991–1.0`.

Conclusion: the typed transaction can work; the unresolved problem is critic/routing quality.

### Slide 20 — Open-ended LLM judging fails

Use Qwen3-8B result:

| Prompt/context | Precision | Recall | F1 | Rare-control FPR |
|---|---:|---:|---:|---:|
| Typed contract | .352 | .190 | .247 | 0 |
| Broad rubric | .185 | .200 | .192 | .25 |
| Temporal pair | **.525** | **.210** | **.300** | .05 |

Failure decomposition:

- age–education recall: `.70–1.00`;
- education FD: at most `.15`;
- spouse consistency: at most `.20`;
- numeric domain and missingness: `0`.

Main finding: the model collapses toward one salient relation. Temporal context improves precision but does not solve relation discovery.

### Slide 21 — Relation routing changes the result

Show the newest completed API runs:

| Focused candidate verification | Precision | Recall | F1 | Rare-control FPR | Tokens | Latency |
|---|---:|---:|---:|---:|---:|---:|
| GPT-4.1-mini | **.989** | .870 | **.926** | 0 | 60,899 | 116 s |
| GPT-5.4-mini | .978 | .870 | .921 | 0 | 61,191 | **68 s** |

Per-relation recall for both models:

- age–education: `1.00`;
- education FD: `1.00`;
- numeric domain: `1.00`;
- spouse consistency: `1.00`;
- missingness coupling: **`.35`**.

Timestep observation:

- GPT-4.1-mini achieved perfect detection at `t≈.034` in this controlled bank;
- performance is not monotonic across timesteps;
- this is not yet an end-to-end sampler result.

### Slide 22 — What the focused result does and does not prove

**Supported**

- Open-ended relation discovery and relation verification are different tasks.
- Routing can eliminate most systematic relation confusion.
- Focused verification passes the preliminary precision gate.

**Not yet supported**

- LLM is better than an executable verifier for these five rules.
- Temporal context improves focused verification.
- Mid-sampling is better than applying the same compiler at `t=0`.
- Results transfer from injected corruption to naturally occurring semantic errors.

Critical caveat: the candidate relation label itself provides useful structure. The research contribution cannot be “give the LLM the rule name.”

---

## Part IV — Refined hypotheses and evaluation

### Slide 23 — Refined hypotheses and kill criteria

**H11 — Record-specific relation routing**

- A cheap router identifies which semantic relation/tool is applicable to each stable preview.
- Success: high coverage with low routing cost; routed verification macro-F1 ≥ `.60`, clean FPR ≤ `.10`.
- Kill: routing requires encoding the full executable rule.

**H12 — Temporal evidence after routing**

- Previous/current previews help distinguish transient artifacts from true violations.
- Kill: focused temporal does not beat focused current-only verification.

**H13 — Verified typed transaction**

- Focused critique compiled into a type-specific action survives denoising and preserves local likelihood/rare modes.
- Success: ≥80% of oracle validity gain with bounded distortion.

**H14 — Mid-sampling advantage**

- Intervention before `t=0` must beat post-repair or best-of-N on a validity–fidelity–cost Pareto frontier.
- Kill: post-repair remains equal or better.

### Slide 24 — Metrics and baselines required for a credible claim

**Validity**

- independent verifier violation rate;
- per-relation precision, recall and calibration;
- action survival and rollback rate.

**Fidelity and coverage**

- marginal KS/TVD;
- pairwise/joint correlation error;
- conditional subgroup fidelity;
- rare-value and rare-pair coverage;
- duplication/diversity.

**Utility**

- TSTR AUROC/F1 or task-specific downstream metrics.

**Cost**

- LLM calls, input/output tokens, latency and GPU time;
- verifier evaluations and denoiser passes.

**Baselines with the same backbone, seeds and action budget**

- no guidance;
- fixed/static rule prior;
- Harpoon-style differentiable guidance;
- CDD-style categorical projection;
- post-repair with the same compiler;
- best-of-N verifier selection;
- oracle routed feedback;
- LLM-routed feedback.

### Slide 25 — Next work and questions for the mentor

Immediate experiments:

1. Complete focused rubric/temporal API runs after fixing control-case handling.
2. Build a router that predicts applicable relation IDs without seeing validity labels.
3. Run one sparse focused critic call at a calibrated late timestep.
4. Apply exactly the same typed compiler at `.10`, `.03`, and `0`.
5. Move beyond Adult hard rules to exception-bearing or ontology/tool-verifiable relations.
6. Evaluate on at least three datasets and three seeds before making a main claim.

Ask the mentor to decide/refine:

- Is the strongest thesis **relation routing**, **verified transactions**, or **exception-safe guidance**?
- Which domain provides semantic ground truth that cannot be reduced to cheap rules?
- Is beating post-repair necessary, or is an improvement in conditional generation/control sufficient?
- Should text–structured consistency become the main domain, or remain future work?

End with the proposed one-sentence thesis:

> We investigate whether sparse, record-specific semantic routing and verified typed transactions can make LLM feedback useful inside mixed-type diffusion, specifically where static rules are incomplete and post-generation repair is insufficient.

---

# Appendix: paper-by-paper cards

Use one compact card per paper if the mentor requests detail. Each card should contain four fixed fields: **gap, method, where the signal enters, implication for this project**.

## A1 — TabSyn

- Gap: unified high-quality synthesis for heterogeneous tables.
- Method: Transformer VAE followed by score diffusion in continuous latent space.
- Signal: no LLM or semantic constraint.
- Implication: decoded `x̂₀` is available, but actions must map back through a learned latent representation.

## A2 — TabDiff

- Gap: direct mixed-type diffusion without VAE information loss.
- Method: numerical and categorical diffusion with feature-specific schedules.
- Signal: classifier-free conditioning for imputation, not semantic LLM feedback.
- Implication: best current backbone for typed intermediate actions.

## A3 — Harpoon

- Gap: control a pretrained tabular diffusion model at inference time.
- Method: differentiable objective evaluated on predicted clean samples; gradients update reverse steps.
- Signal: differentiable loss during denoising.
- Implication: mandatory baseline for executable numeric constraints.

## A4 — GRDM

- Gap: synthesize relational databases without an arbitrary table generation order.
- Method: graph representation based on foreign keys and joint denoising.
- Signal: structural relational graph.
- Implication: possible future extension to multi-table semantic transactions.

## A5 — LLM-TabLogic

- Gap: deterministic semantic relations are inefficiently modeled by diffusion.
- Method: LLM discovers relations and selects compression/decompression structure.
- Signal: before training/sampling.
- Implication: static metadata reasoning and representation selection are already covered.

## A6 — SCTab-Diff

- Gap: weakly specified semantic constraints are missed by distribution matching.
- Method: LLM extracts validated semantic priors; priors condition reverse denoising.
- Signal: static prior throughout denoising.
- Implication: novelty must come from dynamic record/timestep decisions, not prior conditioning alone.

## A7 — LCATS

- Gap: global fidelity does not ensure local logical validity.
- Method: LLM constraint discovery, executable validation, diffusion generation and optimization-based repair.
- Signal: discovery before generation and correction after generation.
- Implication: strongest repair baseline and a challenge to the necessity of inner-loop feedback.

## A8 — Zhao et al.

- Gap: one-shot LLM rule discovery is unreliable.
- Method: executable hypotheses refined using counterexamples, followed by output repair.
- Signal: pre-generation discovery and post-generation repair.
- Implication: relation proposals need empirical validation and explicit counterexamples.

## A9 — SAGE

- Gap: a fixed dependency graph misses value-specific dependencies.
- Method: value-aware pseudo-features and sparse context/logit guidance during autoregressive generation.
- Signal: record-prefix-specific guidance inside LLM decoding.
- Implication: value-conditioned dependencies are not a new claim; diffusion-specific action/reliability remains open.

## A10 — AFT-Tab

- Gap: structured columns and long text become inconsistent when generated separately.
- Method: CTGAN + LLM with discriminator/PPO feedback during training.
- Signal: training-time co-evolution.
- Implication: useful comparison only for a long-text dataset branch.

## A11 — TabDLM

- Gap: unified synthesis of numerical, categorical and textual columns.
- Method: continuous numerical diffusion and masked language diffusion with shared attention.
- Signal: joint denoising, not external critique.
- Implication: suitable future backbone for text–structured consistency, but computationally heavier.

## A12 — PromptLoop

- Gap: fixed prompts cannot correct errors that emerge during diffusion.
- Method: MLLM observes denoised estimates and updates prompts at selected steps.
- Signal: intermediate preview to revised condition.
- Implication: the generic feedback loop is not novel.

## A13 — PPAD

- Gap: early diffusion states are too noisy for reliable semantic feedback.
- Method: lookahead preview, MLLM diagnosis, rollback and resampling.
- Signal: intermediate black-box feedback.
- Implication: motivates readability calibration, abstention and rollback.

## A14 — Free²Guide

- Gap: obtain training-free alignment without gradients through a large evaluator.
- Method: LVLM black-box reward and path-integral/branch guidance.
- Signal: reward during sampling.
- Implication: best-of-N and black-box reward selection are necessary baselines.

## A15 — TTGen

- Gap: test-time scaling for diffusion alignment.
- Method: LVLM prompt revision from clean estimates plus branch selection.
- Signal: intermediate prompt update and external scoring.
- Implication: very close mechanism, but image-only and a workshop paper.

## A16 — Constrained Discrete Diffusion

- Gap: enforce hard constraints in categorical diffusion.
- Method: simplex projection, Gumbel-softmax and augmented Lagrangian updates.
- Signal: constraint projection during discrete denoising.
- Implication: categorical LLM actions must be compared against direct projection.

---

# Presentation guidance

- Put detailed citations in a small footer; do not read them aloud.
- Use `[Executed]`, `[Preliminary]`, and `[Pending]` tags consistently.
- Do not call proxy baselines paper reproductions.
- On every result slide, state sample count, seeds, checkpoint and whether the errors are natural or injected.
- Keep the high focused-verification score next to its caveat: the candidate relation is supplied.
- If time is short, move Slides 5, 8 and 10 to the appendix; keep Slides 3, 6, 7, 9 and 11 for the novelty argument.
