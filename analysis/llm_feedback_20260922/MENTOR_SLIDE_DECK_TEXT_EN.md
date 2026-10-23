# LLM-Guided Feedback During Tabular Diffusion

## Complete slide text for mentor discussion

Recommended length: 20 slides, 30–35 minutes. Everything below is slide-facing English text; short notes in *italics* describe the preferred visual.

---

## Slide 1 — Title

### LLM-Guided Feedback During Tabular Diffusion

**From open-ended critique to verified typed transactions**

Research question:

> Can record-specific semantic feedback improve validity without sacrificing fidelity, rare modes, and cost?

Footer: *Exploratory results and hypotheses for refinement — September 2026*

---

## Slide 2 — Why distributional fidelity is not enough

- A generator may match marginals and pairwise correlations while producing locally inconsistent rows.
- Examples include mismatched education fields, incompatible spouse/sex/marital fields, or inconsistent missing-value sentinels.
- However, plausible relations may have legitimate subgroup exceptions.
- Aggressive guidance can improve validity while erasing rare but valid modes.

### Target trade-off

**Semantic validity ↔ distributional fidelity and utility ↔ exception / rare-mode coverage**

Bottom message:

> A useful controller must improve validity without silently collapsing legitimate variation.

*Visual: three connected circles labelled Validity, Fidelity, Coverage.*

---

## Slide 3 — Related-work landscape

### Where does semantic knowledge enter generation?

| Before sampling | During denoising | After generation |
|---|---|---|
| LLM-TabLogic | Harpoon | LCATS |
| SCTab-Diff | Constrained Discrete Diffusion | Zhao et al. |
| Static relation discovery | PromptLoop / PPAD | Rule or optimization repair |
| Representation or prior | Free²Guide / TTGen | Regeneration |

Second axis:

**Executable rule → differentiable objective → black-box reward → LLM semantic judgment**

Bottom message:

> “LLM + diffusion,” “constraints during denoising,” and “intermediate feedback” are not new by themselves.

---

## Slide 4 — Main tabular directions

### Diffusion backbones

- **TabSyn:** VAE maps mixed types to a continuous latent, followed by score diffusion.
- **TabDiff:** direct numerical and categorical diffusion with feature-specific noise schedules.
- **GRDM:** joint graph diffusion for relational databases.

### LLM knowledge before sampling

- **LLM-TabLogic:** LLM discovers relations and selects compression/decompression structure.
- **SCTab-Diff:** LLM extracts validated semantic priors that condition reverse denoising.

### Post-generation correction

- **LCATS:** LLM constraint discovery, executable validation, and optimization-based repair.
- **Zhao et al.:** executable rule hypotheses refined by counterexamples, then used for repair.

### LLM as generator or joint component

- **SAGE:** value-dependent guidance during autoregressive LLM generation.
- **AFT-Tab / TabDLM:** structured–text generation and joint mixed-modality denoising.

Bottom message:

> The open question is not whether LLM knowledge can be used, but whether dynamic record-level feedback is necessary and cost-effective.

---

## Slide 5 — Lessons from image and video diffusion

- **PromptLoop:** an MLLM observes denoised estimates and updates prompts during sampling.
- **PPAD:** lookahead previews, diagnosis, rollback, and resampling; noisy previews can make feedback harmful.
- **Free²Guide:** black-box LVLM reward provides training-free guidance.
- **TTGen:** LVLM prompt revision plus branch selection at test time.

### Implications for this work

- The generic intermediate-feedback loop is already established.
- Preview readability must be measured rather than assumed.
- Rollback, best-of-N, and post-repair are mandatory baselines.
- A tabular contribution must exploit typed columns, executable checks, exceptions, and rare-mode metrics.

---

## Slide 6 — Defensible novelty boundary

### We cannot claim

- LLM + diffusion.
- Static semantic prior inside denoising.
- Generic constraint guidance.
- Generic intermediate-preview feedback.
- Adaptive timestep scheduling alone.

### Potential contribution

- Record-specific relation or tool routing.
- Calibrated abstention for uncertain and exception-bearing relations.
- Type-specific numerical and categorical transactions.
- Independent verification and rollback before committing an action.
- Direct comparison against the same compiler used as post-repair.

One-sentence gap:

> Can sparse, record-specific semantic routing make intermediate diffusion intervention useful when static rules are incomplete?

---

## Slide 7 — Initial idea and refined architecture

### Initial idea

`x_t → decode x̂₀ → open-ended LLM critique → edit row → continue denoising`

### Refined architecture

`x_t → readable x̂₀ → relation router → focused LLM verifier → compiler → typed action → independent check / rollback → continue`

### LLM output contract

- `violated / satisfied / not_applicable / abstain`
- relation identifier and involved columns
- confidence and compact evidence code
- no arbitrary value writing

### Typed actuator

- numerical projection or bounded update
- categorical remasking or projection
- atomic commit with rollback

*Visual: a seven-block pipeline with the LLM verifier highlighted, followed by a safety gate.*

---

## Slide 8 — Research questions

1. **Readability:** when is `x̂₀(x_t)` stable enough to evaluate?
2. **Need:** which errors cannot be represented by static rules or priors?
3. **Actuation:** which typed actions survive the remaining denoising steps?
4. **Necessity:** does mid-sampling feedback beat post-repair or best-of-N?
5. **Reliability and cost:** can routing and abstention make LLM calls trustworthy and sparse?

Future branch only if supported by data:

6. **Text–structured consistency:** can feedback repair text and structured columns bidirectionally?

---

## Slide 9 — Experimental setup

### Backbone and natural evaluation

- Official TabDiff implementation.
- Adult checkpoint trained for 8,000 epochs.
- 30-step sampler.
- Three seeds × 512 samples for full-checkpoint intervention studies.

### Controlled semantic stress test

- 220 synthetic cases.
- Five semantic relations at five timesteps.
- Corrupted samples, paired-clean controls, and rare/exception controls.
- Only synthetic rows are sent to external APIs.

### Evidence labels

- **Executed:** supports a current conclusion.
- **Preliminary:** small-scale or controlled evidence.
- **Pending:** not used for claims.
- **Proxy baseline:** mechanism tested in the same backbone, not a paper reproduction.

---

## Slide 10 — H1: intermediate previews become readable late

| Timestep `t` | Numerical MAE | Categorical accuracy | Decoded rule violation |
|---:|---:|---:|---:|
| .03 | .0040 | 99.15% | 1.48% |
| .10 | .0114 | 97.37% | 3.30% |
| .25 | .0760 | 92.77% | 13.19% |
| .40 | .2724 | 87.46% | 23.78% |
| .70 | 1.168 | 72.95% | 83.85% |

### Conclusion

- `t≤.10` is the first defensible critic window for this checkpoint.
- Numerical and categorical features stabilize at different rates.
- A universal, arbitrarily selected LLM schedule is not defensible.

*Visual: `outputs/pilot_probe_v2/trajectory_readability.png`.*

---

## Slide 11 — Rule audit: plausibility is not invariance

| Relation | Train / test violation | Interpretation |
|---|---:|---|
| `education → education.num` | 0% / 0% | Hard executable rule |
| spouse → married | .142% / .192% | Near-hard with exceptions |
| `Own-child → age < 30` | 19.69% / 19.66% | Unsafe to hard-enforce |
| `Preschool → ≤50K` | 0% / 3.125% | Train-perfect, test counterexample |

### Implication

- A semantically plausible rule may remove a valid tail subgroup.
- Constraint discovery requires support, held-out counterexamples, and applicability conditions.
- Rare-mode and exception coverage must be primary metrics.

---

## Slide 12 — Strong TabDiff already learns Adult hard rules

| Method | Final hard-rule violation |
|---|---:|
| No guidance | 1.76% |
| One late correction | .20% |
| Targeted recheck | .13% |
| Fixed rule at every step | 0% |
| Post-repair | 0% |

### Conclusion

- Hard-rule enforcement on Adult is not a convincing LLM use case.
- Static guidance and repair are already strong and cheap.
- Evaluation must move to exception-bearing, higher-order, or externally grounded semantics.

*Visual: `slides/figures/full_checkpoint_baselines.png`.*

---

## Slide 13 — Typed actuation matters more than repeated critique

| Action | Final hard-rule violation |
|---|---:|
| No guidance | 1.56% |
| `x̂₀` correction at `.03` | .26% |
| State projection at `.03` | .13% |
| State projection at `.10` | **.065%** |
| |
| Post-repair | 0% |

### Interpretation

- `x̂₀`-only numerical corrections can be attenuated by later denoising.
- Direct typed state projection is much more durable.
- Rechecking adds little once the actuator is correct.
- Mid-sampling still has no demonstrated Pareto advantage over repair.

*Visual: `slides/figures/actuator_comparison.png`.*

---

## Slide 14 — Controlled stress test isolates the mechanism

### Five injected relations

- education functional dependency
- numerical domain validity
- spouse consistency
- age–education plausibility
- missingness coupling

### End-to-end mechanism result

| Branch | Intended final violation |
|---|---:|
| Corruption without feedback | 100% |
| Oracle rollback | 0% |
| Static-verifier rollback | 0% |

- Categorical agreement with the clean trajectory after rollback: approximately `.991–1.0`.
- The transaction mechanism can work.
- Critic selection and the necessity of mid-sampling intervention remain unresolved.

Important caveat: this is a controlled corruption stress test, not a natural error-prevalence estimate.

---

## Slide 15 — Open-ended LLM judging fails

### Qwen3-8B over 220 controlled cases

| Prompt/context | Precision | Recall | F1 | Rare-control FPR |
|---|---:|---:|---:|---:|
| Typed contract | .352 | .190 | .247 | 0 |
| Broad semantic rubric | .185 | .200 | .192 | .25 |
| Temporal pair | **.525** | **.210** | **.300** | .05 |

### Failure pattern

- Age–education recall: `.70–1.00`.
- Education FD: at most `.15`.
- Spouse consistency: at most `.20`.
- Numerical domain and missingness: `0`.

Conclusion:

> The model collapses toward one salient relation. Temporal context improves precision but does not solve relation discovery.

---

## Slide 16 — Relation routing changes the result

| Focused candidate verification | Precision | Recall | F1 | Rare-control FPR | Tokens | API latency |
|---|---:|---:|---:|---:|---:|---:|
| GPT-4.1-mini | **.989** | .870 | **.926** | 0 | 60,899 | 116 s |
| GPT-5.4-mini | .978 | .870 | .921 | 0 | 61,191 | **68 s** |

### Per-relation recall

- education FD: `1.00`
- numerical domain: `1.00`
- spouse consistency: `1.00`
- age–education plausibility: `1.00`
- missingness coupling: **`.35`**

Main result:

> Open-ended relation discovery and focused relation verification are fundamentally different tasks.

*Visuals: `slides/figures/open_vs_focused_llm.png` and `focused_relation_recall.png`.*

---

## Slide 17 — What this result proves — and what it does not

### Supported

- Routing removes most systematic cross-relation confusion.
- Focused verification passes the preliminary precision gate.
- A smaller/cheaper API model is already competitive with a larger mini model.

### Not yet supported

- LLM is better than an executable verifier for these five relations.
- Temporal context improves verification after routing.
- Mid-sampling is better than applying the same compiler at `t=0`.
- Injected errors represent naturally occurring semantic failures.

### Critical caveat

> The candidate relation label provides useful structure. “Give the LLM the rule name” is not a research contribution.

---

## Slide 18 — Refined hypotheses and kill criteria

### H11 — Record-specific relation routing

A cheap router identifies which semantic relation or external tool is applicable to each stable preview.

- Success: routed verification macro-F1 ≥ `.60`, clean FPR ≤ `.10`.
- Kill: routing requires encoding the complete executable rule.

### H12 — Temporal evidence after routing

Previous/current previews distinguish transient artifacts from persistent violations.

- Kill: focused temporal does not outperform focused current-only verification.

### H13 — Verified typed transaction

A compiled action recovers at least 80% of the oracle validity improvement with bounded distortion.

- Kill: excessive rollback, distortion, or rare-mode loss.

### H14 — Mid-sampling advantage

The transaction must beat post-repair or best-of-N on a validity–fidelity–cost Pareto frontier.

- Kill: the same compiler at `t=0` is equal or better.

---

## Slide 19 — Evaluation protocol for a credible claim

### Metrics

- **Validity:** independent verifier violation, per-relation precision/recall, calibration.
- **Action quality:** survival, rollback, corrected columns, local distortion.
- **Fidelity:** marginal KS/TVD, joint correlation error, conditional subgroup fidelity.
- **Coverage:** rare values, rare pairs, exceptions, diversity, duplication.
- **Utility:** TSTR AUROC/F1 or domain-specific downstream tasks.
- **Cost:** calls, tokens, latency, GPU time, denoiser and verifier evaluations.

### Matched baselines

- no guidance
- static prior / fixed rule
- Harpoon-style differentiable guidance
- categorical projection
- post-repair with the same compiler
- best-of-N verifier selection
- oracle dynamic feedback
- LLM-routed feedback

All comparisons must use the same backbone, split, seeds, sample count, compiler, and action budget.

---

## Slide 20 — Next work and questions for discussion

### Immediate work

1. Complete focused rubric and focused temporal ablations.
2. Build a router that predicts relation/tool applicability—not validity.
3. Run one sparse focused critic call at a calibrated late timestep.
4. Apply the same typed compiler at `.10`, `.03`, and `0`.
5. Move from Adult hard rules to semantics requiring an ontology, external tool, higher-order context, or text–structured reasoning.
6. Validate on at least three datasets and three seeds.

### Questions for the mentor

- Should the central thesis be relation routing, verified transactions, or exception-safe guidance?
- Which domain provides independent semantic ground truth that cannot be reduced to cheap rules?
- Must the method beat post-repair, or is improved conditional control sufficient?
- Should text–structured consistency become the main use case?

### Proposed thesis

> We investigate whether sparse, record-specific semantic routing and verified typed transactions can make LLM feedback useful inside mixed-type diffusion, specifically where static rules are incomplete and post-generation repair is insufficient.

---

## Compact reference slide

- TabSyn — ICLR 2024; TabDiff — ICLR 2025; GRDM — NeurIPS 2025.
- Harpoon — ICLR 2026; Constrained Discrete Diffusion — NeurIPS 2025.
- LLM-TabLogic — preprint; SCTab-Diff — 2026 preprint.
- LCATS — KDD 2026; Zhao et al. — 2026 preprint.
- SAGE and AFT-Tab — ACL 2026; TabDLM — 2026 preprint.
- PromptLoop — CVPR 2026; PPAD — preprint.
- Free²Guide — ICCV 2025; TTGen — CVPR 2025 workshop.

Full URLs and source-status notes are available in `REPORT.md`.
