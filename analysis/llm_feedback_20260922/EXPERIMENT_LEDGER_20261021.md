# Experiment ledger — 2026-10-21

## PS-DFR smoke validation

- Slurm job: `1108989`
- Script: `slurm/809_psdfr_smoke.sbatch`
- Purpose: compile the new posterior router and run finite/shape/frozen-membership invariants on an allocated CPU node.
- Success marker: `PSDFR_SMOKE_OK`
- Logs: `logs/psdfr-smoke-1108989.{out,err}`

## News posterior-routing screen

- Slurm array job: `1108990` (`0-1%2`)
- Dependency: `afterok:1108989`
- Script: `slurm/810_news_posterior_routing_screen.sbatch`
- Repository revision at submission: `5ecdb33` plus pre-existing uncommitted user changes; exact diff hash is recorded by each task.
- Hypothesis: posterior-marginalized, row-specific factor routing can preserve the stable one-shot gains while reducing hard-bin boundary errors; recomputing membership across pulses tests genuinely dynamic routing.
- Paired screening seeds: `20261240`, `20261241`
- Variants, run sequentially on each allocated node:
  - `hard_one_shot`
  - `hard_two_pulse`
  - `global_two_pulse`
  - `frozen_soft_two_pulse`
  - `dynamic_soft_two_pulse`
- Fixed budget: same News checkpoint/config/typed graph/trust geometry, 8,192 samples, 50 sampler steps; two-pulse variants use matched strength and pulse budget.
- Primary comparison: public News Shape, Trend, Alpha, Beta, R2, and NRMSE against the paired backbone.
- Routing diagnostics: posterior entropy, max posterior probability, bin-switch fraction, movement magnitude, changed rows/cells, transport calls, trace, runtime, source hashes, checkpoint stat, GPU metadata, and unconditional sample hashes.
- Output root: `outputs/news_psdfr_screen_s${SEED}_1108990_${SLURM_ARRAY_TASK_ID}`
- Logs: `logs/news-psdfr-1108990_{0,1}.{out,err}`
- Advancement rule: do not broaden the grid; confirm only a variant that improves the main metrics consistently across paired seeds and remains mechanistically distinct from the global comparator.

## Stopped branch

- Current Adult categorical hybrid was not advanced: the three-seed confirmation (`1100253`) did not reproduce the earlier broad win despite nonzero guidance activity.

## News uncertainty-aware router funnel

- Smoke job: `1109015` (`COMPLETED`, exit `0:0`).
- Wide-screen array: `1109016` (`0-11%12`), one GPU per task.
- Selection job: `1109017`, dependency `afterok:1109016`.
- Hypothesis: ambiguous target-bin posteriors should receive a weak global fallback or abstain; bin changes should require a posterior advantage; a global 5% tail reserve should reduce the Beta/coverage loss without erasing Shape/Trend gains.
- Locked paired reference: dynamic soft quantile routing, four bins, temperature `.75`, two pulses of `.05`, gap `.18`.
- Six screened candidates, each on seeds `20261250` and `20261251`:
  - `sharp035`
  - `confidence55_global025`
  - `confidence55_abstain`
  - `switchmargin15`
  - `confidence55_switchmargin15`
  - `confidence55_switchmargin15_tail05`
- Every task runs the reference and candidate sequentially with identical checkpoint, seed, base noise, typed graph, projection/trust geometry, 8,192 rows and 50 diffusion steps.
- Promotion is fail-closed and performed inside Slurm. At most two candidates can advance; missing seeds, incomplete metrics, inconsistent Shape/Trend, or Beta/utility downside prevents promotion.
- R1 output pattern: `outputs/news_uar_r1_<variant>_s<seed>_1109016_<task>`.
- Selection output pointer: `outputs/news_uar_r1_selection_1109016.pointer`.
- Prepared confirmation script: `slurm/814_confirm_news_uar_winners.sbatch`; it runs Backbone, Global, Frozen, locked Reference and each selected candidate on six fresh seeds, with at most 12 concurrent GPUs.

### R1 decision

- All 12 tasks and selector completed successfully.
- Selection root: `outputs/news_uar_r1_selection_1109016_1109017`.
- No arm passed the predeclared promotion rule; dispatch job `1110338` wrote `status=no_promotion` and did not allocate confirmation GPUs.
- Best mechanistic signal: `switchmargin15` was effectively neutral on Shape/Trend (`+0.000060`, `+0.000368` mean versus locked reference) while improving R2 by `+0.022119` and NRMSE benefit by `+0.007073`; this was judged too small/inconsistent to confirm.
- Confidence fallback improved some Alpha/Beta cells but reduced Shape/Trend, so threshold refinement was stopped rather than expanded into another local grid.

## News adaptive-resolution posterior routing

- Smoke job: `1110351`.
- R2 array: `1110352` (`0-11%12`), one GPU per task.
- Selection job: `1110353`, dependency `afterok:1110352`.
- Hypothesis: fixed q4 conditioning is a bias/variance compromise that varies by row; posterior confidence can route uncertain rows to q2, intermediate rows to q4, and confident rows to q8 while keeping a convex action budget.
- Locked reference: soft q4 one-shot, strength `.10`, middle window `.20-.35`.
- Six paired candidates on seeds `20261260` and `20261261`:
  - `fixed_q2_one`
  - `fixed_q8_one`
  - `fixed_q4_two`
  - `adaptive_248_one`
  - `adaptive_248_two`
  - `adaptive_248_two_tail05`
- Adaptive resolution thresholds are fixed before results (`.52`/`.68`); tail arm reserves the globally rarest 5% of rows only after the q2/q4/q8 delta blend.
- R2 output pattern: `outputs/news_arr_r2_<variant>_s<seed>_1110352_<task>`.
- Selection output pointer: `outputs/news_arr_r2_selection_1110352.pointer`.

### R2 decision

- Smoke, all 12 GPU tasks, and selector completed successfully (`0:0`).
- Selection root: `outputs/news_arr_r2_selection_1110352_1110353`.
- No arm passed the predeclared promotion rule, so no confirmation GPUs were allocated.
- `fixed_q2_one` ranked first: mean deltas versus its paired locked reference were Shape `+0.001598`, Trend `+0.000059`, Alpha `+0.006510`, Beta `-0.017497`, R2 `-0.008122`, and NRMSE benefit `-0.002082`. It improved only 3/6 mean metrics and Trend won only 1/2 seeds.
- `adaptive_248_two` produced the largest mean Shape gain (`+0.007868`) but was negative on Trend (`-0.000336`), Alpha (`-0.045166`), Beta (`-0.030518`) and R2 (`-0.046324`), so it was not promoted.
- Adaptive-resolution/threshold refinement is stopped. The next branch is a train-only cross-fitted factorized residual witness with row-by-factor routing, bounded middle-step action, target read-only constraints, and explicit critic-exploitation diagnostics.

## Cross-fitted factorized residual witness (CFRW)

- CPU/sklearn smoke job: `1110602`.
- Paired cross-dataset screen: `1110603` (`0-7%8`), dependency `afterok:1110602`.
- CPU selector: `1110604`, dependency `afterok:1110603`.
- Design: News and Default crossed with `W0`, `W4`, `W8`, and `W12`; every GPU task runs its locked `G` reference and candidate sequentially on the same allocation and validates identical unconditional SHA-256 locally.
- News seed: `20261340`; Default seed: `20261341`; 8,192 rows and 50 sampler steps.
- `W0` contains marginal residual heads only. `W4/W8/W12` add target-conditional and up to 4/8/12 pair residual heads selected by OOF reliability.
- Action budget is fixed before results: one pulse, strength `.10`, interval `.20-.35`, one factor per row, top 15% rows per factor, score threshold `.25`, head-local clip `2.0`; the existing encoded trust region remains active.
- Target is a read-only predictor and never an intervention destination. Conditional heads are fitted from train-only out-of-fold residuals; heldout/test rows and evaluator metrics are excluded from fitting.
- Promotion is fail-closed across both datasets and checks paired Shape/Trend gains versus both `B` and `G`, Alpha/Beta/utility floors, runtime, target preservation, retained head families, selected rows/cells, exact call count, and local unconditional hashes.
- Output pattern: `outputs/cfrw_screen_<dataset>_<variant>_1110603_<task>`.
- Selection pointer after completion: `outputs/cfrw_selection_1110603.pointer`.

### CFRW integration correction

- Smoke `1110602` completed successfully (`CFRW_SMOKE_OK`).
- In the first screen `1110603`, News `W0` completed but `W4/W8/W12` stopped fail-closed because no target-only head reached the predeclared OOF `R2 >= .02`; this was a model-selection fact, not a sampler/import failure. The incomplete round and selector `1110604` were cancelled.
- The treatment was corrected from `marginal_target_pair` to `marginal_pair`; the target remains excluded from destinations. No strength/window/routing threshold was changed.
- Corrected paired screen: `1110696` (`0-7%8`). Corrected selector: `1110699`.
- Corrected output pattern: `outputs/cfrw_screen_<dataset>_<variant>_1110696_<task>`; selection pointer: `outputs/cfrw_selection_1110696.pointer`.

## Factorized RFF-MMD guidance

- Motivation: borrow the distribution-mismatch objective of MMD guidance while replacing a global batch action with typed marginal/pair random-feature witnesses and sparse row-by-factor routing on the clean middle-step preview.
- CPU smoke: `1110712`.
- News paired screen: `1110713` (`0-3%4`), dependency `afterok:1110712`.
- CPU selector: `1110721`, dependency `afterok:1110713`.
- Arms on seed `20261350`: `mmd_marginal_one`, `mmd_pair4_one`, `mmd_pair8_one`, `mmd_pair8_two`. Every task runs the same locked q4 `G` reference and candidate sequentially and checks unconditional hashes locally.
- Fixed operator: 64 random Fourier features/head, train-only reference mean and robust scaling, median-bandwidth RBF, top one factor per row, top 15% rows per factor, score floor `.25`, standardized action clip `.50`, existing encoded trust region.
- One-pulse arms use strength `.10` in `.20-.35`; the two-pulse arm uses `.05 + .05` in `.10-.40` with minimum gap `.18`.
- Promotion requires positive paired Shape and Trend versus both `B` and `G`, utility/coverage floors, positive predicted MMD decrease, nonzero sparse action, target preservation, correct head budget/call count, and runtime no more than `2.5x` its paired `G`.
- Output pattern: `outputs/news_frmmd_<variant>_1110713_<task>`; selection pointer: `outputs/news_frmmd_selection_1110713.pointer`.

### CFRW corrected-round decision

- Corrected screen `1110696` and selector `1110699` completed successfully; all eight task-local unconditional pairs matched exactly.
- Selection root: `outputs/cfrw_selection_1110696_1110699`; no arm was promoted.
- `W8` ranked first. It beat its file-local backbone on mean Shape (`+0.009731`) and Trend (`+0.018830`) across News+Default, but remained below the locked `G` on mean Shape (`-0.034887`) and Trend (`-0.006668`) and lost Alpha (`-0.151479`).
- `W4` exposed a complementary trade-off on News: versus `G`, R2 improved `+0.083142` and NRMSE benefit improved `+0.029977`, while Shape/Trend fell `-0.027786/-0.013355` and Beta fell `-0.234375`.
- Standalone CFRW is stopped. Its complementary utility direction motivates bounded fusion with the locked `G`, not another witness strength sweep.

### Factorized RFF-MMD decision

- Smoke `1110712`, all four paired GPU tasks in `1110713`, and selector `1110721` completed successfully.
- Selection root: `outputs/news_frmmd_selection_1110713_1110721`; no arm was promoted.
- Best arm `mmd_pair8_two` remained below `G` on Shape (`-0.023884`), Trend (`-0.011498`), R2 (`-0.177742`) and NRMSE benefit (`-0.054586`).
- The mechanism worked as specified (positive predicted MMD decrease and sparse routed action), so the failure is objective misalignment with the public metrics rather than an inactive implementation. Standalone factor-MMD is stopped.

## Reference-preserving witness consensus

- Motivation: preserve the full locked `G` action, then add only a small CFRW component after action-space conflict surgery. This directly targets the observed main-metric/utility trade-off and borrows the projection principle of PCGrad without claiming that generic gradient surgery is novel.
- CPU smoke: `1110847`.
- Paired cross-dataset screen: `1110859` (`0-7%8`), dependency `afterok:1110847`.
- CPU selector: `1110862`, dependency `afterok:1110859`.
- Arms: coordinate-aligned fusion at `.15/.30` reference-norm budget (`aligned15`, `aligned30`) and row-vector PCGrad fusion at `.15/.30` (`pcgrad15`, `pcgrad30`).
- Locked `G` is unchanged: soft q4 one-shot for News and the locked categorical/numerical hybrid for Default. The witness is the cross-fitted marginal+four-pair CFRW; target follows the locked reference exactly.
- Seeds: News `20261360`, Default `20261361`; 8,192 rows, 50 steps. Each GPU task runs `G` and candidate sequentially and validates its local unconditional hash.
- Output pattern: `outputs/consensus_<dataset>_<variant>_1110859_<task>`; selection pointer: `outputs/consensus_selection_1110859.pointer`.

### Consensus decision

- Smoke `1110847`, all eight paired GPU tasks in `1110859`, and selector `1110862` completed successfully; no arm was promoted.
- Selection root: `outputs/consensus_selection_1110859_1110862`.
- `aligned30` was the least-negative fusion: averaged across News+Default it changed Shape by `+0.000029`, Trend by `+0.000678`, and Alpha by `+0.002441` versus locked `G`, but it lost Default AUC by `-0.007897` and did not improve Default Shape.
- On News, `aligned30` improved Shape/Trend only `+0.000058/+0.000738` while reducing R2 by `-0.028460` and NRMSE benefit by `-0.008150`. PCGrad arms did not recover the utility loss.
- The witness-consensus branch is stopped: action-space surgery is active but adds negligible public-metric gain and no robust cross-dataset improvement.

## Default hybrid SOTA local search

- Paired GPU array: `1110907` (`0-11%12`), one GPU per task.
- CPU selector: `1110908`, dependency `afterok:1110907`.
- Goal: directly tune the strongest existing numerical copula + categorical marginal/target hybrid on Default before committing to another new mechanism.
- Locked `G`: numeric conditional copula strength `.50`, tail preservation `.10`, one application in `.10-.40`; categorical marginal-target strength `.25`, marginal scale `.10`, pair scale `2`, window `.20-.50`.
- Twelve one-coordinate/joint arms: `num035`, `num065`, `cat015`, `cat035`, `marg005`, `marg020`, `pair150`, `pair300`, `tail005`, `tail015`, `joint_low`, `joint_high`.
- Fixed screen: seed `20261370`, 8,192 rows, 50 sampler steps. Every task runs locked `G` then its candidate on the same allocated GPU and requires identical unconditional SHA-256 before evaluation.
- Promotion is paired and fail-closed: positive Shape and Trend versus `G`, bounded Alpha/Beta/AUC/DCR downside, runtime at most `1.5x`, at most two winners. TabSyn local numbers are retained only as one-run gap context, not as paired selection evidence.
- Output pattern: `outputs/default_sota_<variant>_1110907_<task>`; selection pointer: `outputs/default_sota_selection_1110907.pointer`.

### Local-search decision and confirmation

- All twelve paired tasks in `1110907` and selector `1110908` completed successfully. Selection root: `outputs/default_sota_selection_1110907_1110908`.
- `joint_high` ranked first versus its same-node `G`: Shape `+0.010133`, Trend `+0.000134`, Alpha `+0.006510`, Beta `+0.069987`, and AUC `+0.002395` (all benefit-oriented).
- `pair300` ranked second: Shape `+0.006177`, Trend `+0.005402`, Alpha `+0.025031`, Beta `-0.056966`, and AUC `-0.000016`.
- The two selected configurations are now locked; no post-selection parameter edits are permitted in confirmation.
- Six-fresh-seed paired confirmation array: `1110925` (`0-11%12`), seeds `20261371..20261376`, one candidate/reference pair per allocated GPU.
- CPU fail-closed confirmation summary: `1110926`, dependency `afterok:1110925`.
- Confirmation requires mean Shape and Trend benefit versus both `G` and `B`, at least `4/6` Shape wins and `4/6` Trend wins versus `G`, bounded Alpha/Beta/AUC/DCR downside, and mean runtime at most `1.5x G`.
- Confirmation output pattern: `outputs/default_sota_confirm_<variant>_s<seed>_1110925_<task>`; summary pointer: `outputs/default_sota_confirmation_1110925.pointer`.

### Six-seed confirmation decision

- All twelve paired tasks in `1110925` and summary job `1110926` completed successfully. Summary root: `outputs/default_sota_confirmation_1110925_1110926`.
- Neither arm passed the full predeclared confirmation rule.
- `joint_high` produced the strongest robust Shape result: mean Shape benefit versus paired `G` `+0.004346`, wins `5/6`, bootstrap 95% interval `[+0.001135, +0.007341]`, and mean raw Shape `1.429875` versus the local one-run TabSyn context `1.4810`.
- `joint_high` did not close Trend: mean Trend benefit versus `G` was only `+0.000342`, its mean Trend benefit versus `B` was `-0.001052`, and mean raw Trend was `1.676550` versus TabSyn context `1.6499`.
- `pair300` improved mean Shape/Trend versus `G` by `+0.002860/+0.001041`, but Trend won only `3/6` seeds and both primary bootstrap intervals crossed zero.
- Decision: retain `joint_high` as a Shape-specialized candidate, but do not claim a comprehensive SOTA win. The next branch targets set-level pair-trend correction without editing generated rows.

## Train-only balanced particle / coreset resampling

- Method: generate a 2x locked-`G` particle pool, estimate optional train/pool density ratios, rake weights toward train-only marginal and selected pair moments, draw without replacement, and select the particle set with the lowest predeclared train-only moment discrepancy. No test/heldout rows are read and no generated row is edited.
- CPU integration smoke `1110937` completed successfully for both logistic and XGBoost density-ratio paths. Both outputs had unique selected indices and lower moment objective than their internal uniform control.
- Default screen array: `1110943` (`0-7%8`), one GPU per task; CPU selector: `1110944`, dependency `afterok:1110943`.
- Pool/selection budget: 16,384 locked-`G` rows -> 8,192 unique selected rows, seed `20261380`, 50 diffusion steps, 32 candidate without-replacement draws.
- Arms: `uniform32`, `rake_marg35`, `rake_pair35`, `rake_pair60`, `ratio_logit50`, `ratio_xgb50`, `hybrid_logit50`, `hybrid_xgb15`.
- Every task evaluates four equal-sized tables: backbone `B`, first-half locked `G`, a uniform pool subset `R`, and the selected candidate. Promotion requires Shape/Trend gains versus all `B/G/R`, bounded utility/privacy downside, positive internal moment-objective gain, unique selected indices, and complete provenance.
- Output pattern: `outputs/default_particle_<variant>_1110943_<task>`; selection pointer: `outputs/default_particle_selection_1110943.pointer`.

### Particle screen decision and confirmation

- All eight GPU tasks in `1110943` and selector `1110944` completed successfully. Selection root: `outputs/default_particle_selection_1110943_1110944`.
- `ratio_xgb50` ranked first: versus paired first-half `G`, Shape `+0.354234`, Trend `+0.413008`, Alpha `-0.018156`, Beta `+0.449219`, AUC `-0.002366`; raw Shape/Trend `1.202033/1.248444`.
- `hybrid_logit50` ranked second: versus `G`, Shape `+0.305906`, Trend `+0.015262`, Alpha `+0.669478`, Beta `+1.535645`, AUC `-0.002857`.
- The simple `uniform32` ablation also passed all gates with Shape/Trend `+0.251740/+0.280757` and AUC `+0.002329` versus `G`, supporting that set-level best-of-N balanced selection—not only the density-ratio classifier—drives the gain.
- All five eligible candidates beat the local one-run TabSyn Shape context; the strongest arms also beat its Trend context. These are screening observations, not confirmation claims.
- Locked six-fresh-seed confirmation array: `1110956` (`0-11%12`), seeds `20261381..20261386`, one GPU per task. CPU summary: `1110957`, dependency `afterok:1110956`.
- Confirmation requires positive mean Shape/Trend versus `B`, first-half `G`, and uniform pool control `R`; at least `5/6` wins for both primary metrics versus both `G` and `R`; bounded Alpha/Beta/AUC/DCR downside; unique selected indices; and positive internal moment-objective gain on every seed.
- Confirmation output pattern: `outputs/default_particle_confirm_<variant>_s<seed>_1110956_<task>`; summary pointer: `outputs/default_particle_confirmation_1110956.pointer`.
- Multi-seed mechanism ablation array `1110971` (dependency `afterok:1110956`) reuses the six `ratio_xgb50` confirmation pools without rerunning diffusion. It compares `B`, first-half `G`, random subset `R`, `uniform32`, `rake_marg35`, and `ratio_xgb50` on identical per-seed pools.
- Ablation summary job: `1110972`, dependency `afterok:1110971`; output pointer: `outputs/default_particle_ablation_1110971.pointer`.

### First particle confirmation decision

- All twelve tasks in `1110956` and summary `1110957` completed successfully. Summary root: `outputs/default_particle_confirmation_1110956_1110957`.
- Both selected arms reproduced very large primary gains but failed the predeclared utility floors, so neither was declared confirmed.
- `ratio_xgb50`: Shape/Trend wins `6/6` and `6/6`, mean benefits versus `G` `+0.355075/+0.261566`, raw means `1.034745/1.445295`; rejected because Beta benefit was `-0.279541 < -0.20` and AUC benefit was `-0.005256 < -0.005`.
- `hybrid_logit50`: Shape/Trend wins `6/6` and `6/6`, mean benefits `+0.343285/+0.154674`, raw means `1.111730/1.518697`; rejected because Beta benefit was `-0.279948 < -0.20`.
- No thresholds were relaxed after seeing results.

### Multi-seed mechanism ablation decision

- All six pool-reuse tasks in `1110971` and summary `1110972` completed successfully. Summary root: `outputs/default_particle_ablation_1110971_1110972`.
- `uniform32` isolated set-level best-of-32 selection: mean Shape/Trend `1.256672/1.560262`, benefits versus `G` `+0.133148/+0.146599`, Shape wins `6/6`, Trend wins `5/6`, and AUC benefit `+0.000093`.
- `rake_marg35` was the strongest utility-preserving trade-off: mean Shape/Trend `1.103346/1.541485`, benefits `+0.286474/+0.165376`, wins `6/6` on both, Alpha `+0.224156`, Beta `-0.006510`, and AUC `+0.004419` versus `G`.
- `ratio_xgb50` remains the strongest primary-only arm but confirms the density-ratio utility trade-off. The comparison supports distinct contributions from oversampling/random control, set-level moment search, marginal raking, and classifier density ratio.
- `rake_marg35` is now locked for a fresh-pool confirmation; no parameter was changed from its original screen/ablation definition.
- Fresh confirmation array: `1110985` (`0-5%6`), seeds `20261391..20261396`; CPU summary: `1110986`, dependency `afterok:1110985`.
- Fresh confirmation output pattern: `outputs/default_rake_confirm_s<seed>_1110985_<task>`; summary pointer: `outputs/default_rake_confirmation_1110985.pointer`.

### Fresh-pool rake decision

- All six tasks in `1110985` and summary `1110986` completed successfully. Summary root: `outputs/default_rake_confirmation_1110985_1110986`.
- The locked `rake_marg35` arm did not pass the predeclared confirmation rule. It retained a robust Shape benefit versus `G` of `+0.250130` with `6/6` wins and bootstrap 95% interval `[+0.207790, +0.290739]`; mean raw Shape was `1.108579`.
- Trend benefit versus `G` was only `+0.049995`, with `3/6` wins and interval `[-0.054202, +0.164686]`. Beta benefit was `-0.330675`, below the locked `-0.20` floor. No threshold was relaxed.
- Decision: retain marginal raking as a Shape-specialist ablation, but do not call it a comprehensive SOTA result.

## Train-only Pareto particle selection

- Next hypothesis: preserve marginal raking while decoupling the final without-replacement set objective. The new selector jointly ranks univariate discrepancy, selected pair discrepancy, target-feature pair discrepancy, and a rare-cell under-coverage penalty. It does not read test/heldout rows or evaluator outputs, and it never edits generated rows.
- The screen reuses the six already-generated fresh pools from `1110985`; no diffusion generation is repeated. Eight configurations are locked before evaluation: `pw075`, `pw150`, `pw300`, `cov150`, `tail20`, `temp50`, `target16`, and `uniform150`.
- Each configuration uses 128 candidate subsets of 8,192 rows from the same 16,384-row pool. Target-pair quotas and rare-cell coverage weights are explicit in per-run provenance and `selection_metadata.json`.
- Promotion remains fail-closed across all six seeds: positive mean Shape/Trend versus `B/G/R`, at least `5/6` wins versus both `G` and `R`, unchanged Alpha/Beta/AUC/DCR floors, and a positive train-only selection-objective gain on every seed. At most two arms can advance to new-pool confirmation.
- Syntax smoke job `1111654` completed successfully. Multi-seed GPU screen: `1111655` (`0-47%12`); dependent CPU selector: `1111656`. Output pattern: `outputs/default_pareto_<variant>_s<seed>_1111655_<task>`; selection pointer: `outputs/default_pareto_selection_1111655.pointer`.
- A bounded one-shot confirmation dispatcher is queued as `1111698`, dependent on the selector and confirmation-summarizer syntax smoke `1111695`. If zero candidates pass it writes `status=no_promotion`; otherwise it submits exactly one six-seed fresh-pool array and one dependent summary. Each fresh-seed task generates one pool and evaluates all selected arms, avoiding duplicate diffusion runs.

### Pareto-particle decision

- All 48 tasks in `1111655` and selector `1111656` completed successfully. Selection root: `outputs/default_pareto_selection_1111655_1111656`. No arm passed every locked gate; dispatcher `1111698` wrote `status=no_promotion` and allocated no fresh-pool confirmation GPUs.
- `temp50` gave the largest primary gains (Shape/Trend `+0.422188/+0.146706`, Trend wins `5/6`) but failed Beta (`-0.433214`). `tail20` preserved Beta (`+0.098741`) with Shape/Trend `+0.257267/+0.076756`, but Trend won only `4/6`. `uniform150` was utility-neutral with Shape/Trend `+0.148150/+0.120585`, also with only `4/6` Trend wins.
- No utility floor or win-count requirement was relaxed. The common failure on a small number of seeds motivates stability aggregation rather than another scalar-weight grid.

## Stability-aggregated particle coresets

- New hypothesis: treat independently useful subset selectors as noisy voters over the same particle pool. Aggregate row membership from the locked `G` first half, `tail20`, `uniform150`, and optionally `temp50`; then marginally rake the vote prior and perform the same train-only pair/coverage best-of-128 set search.
- This is a bounded stability-selection/coreset branch, not metric-based ensembling: no evaluator score, heldout row, or test row enters the aggregation. Four locked arms vary only voter inclusion and consensus concentration: `gt4`, `gtu2`, `gtu4`, `gttu3`.
- Syntax smoke: `1111708`; 24-task selection/evaluation screen at up to 12 GPUs: `1111709`; dependent fail-closed selector: `1111710`. The screen reuses the six existing pools and all recorded row-index provenance.

### Stability-screen decision and fresh confirmation

- All 24 tasks in `1111709` and selector `1111710` completed successfully. Selection root: `outputs/default_stability_selection_1111709_1111710`.
- `gt4` passed every locked screen gate: mean Shape/Trend benefits versus `G` `+0.323067/+0.136149`, Shape/Trend wins `6/6` and `5/6`, Alpha `+0.301771`, Beta `-0.039605`, AUC `+0.004649`; raw mean Shape/Trend `1.035643/1.403361`.
- `gt4` is the locked consensus of the `G` first half (vote weight 1) and train-only `tail20` selection (vote weight 2), with consensus concentration 4, marginal raking, and pair/coverage best-of-128 selection. No post-screen hyperparameter was changed.
- Fresh-pool syntax smoke: `1111734`; six-seed GPU confirmation array: `1111735` (`0-5%6`, seeds `20261411..20261416`); dependent CPU summary: `1111736`. Each task generates a new 16,384-row pool, rebuilds the `tail20` voter without test access, constructs `gt4`, and evaluates against `B/G` plus an independently fixed random subset `R`.

### Stability fresh-pool decision

- All six tasks in `1111735` and summary `1111736` completed successfully. Summary root: `outputs/default_gt4_confirmation_1111735_1111736`.
- `gt4` reproduced extremely strong primary gains: Shape `+0.459602` (`6/6`, bootstrap 95% interval `[+0.385370,+0.538063]`) and Trend `+0.164424` (`5/6`, interval `[+0.051482,+0.303100]`) versus `G`; raw means were `0.981723/1.492174`.
- It did not pass the comprehensive rule because Beta benefit was `-0.428060 < -0.20`. Alpha was `+0.403912` and AUC was `-0.001941`, so manifold recall—not global precision or downstream AUC—is now the isolated failure mode. No threshold was relaxed.
- Decision: retain `gt4` as the strongest confirmed primary-metric/Shape candidate, but not a comprehensive SOTA claim.

## Train-only geometric coverage constraint

- New mechanism: encode real-train and pool rows in mixed-type min-max/one-hot geometry, compute each train row's real 2-nearest-neighbor radius and its 16 nearest pool particles, then require candidate sets to preserve the locked `G` train-coverage proxy. The selection remains train-only and never reads evaluator metrics, heldout rows, or test rows.
- Four locked penalty/floor configurations are evaluated on the six fresh pools from `1111735`: `geo1_t0`, `geo5_t0`, `geo20_t0`, and `geo5_t005`. They share the exact locked `gt4` proposal distribution; only the train-coverage constraint penalty/tolerance differs.
- Syntax smoke: `1111743`; 24-task GPU screen at up to 12-way concurrency: `1111744`; dependent fail-closed selector: `1111745`. Promotion additionally requires zero train-coverage deficit on every seed, on top of the unchanged Shape/Trend win and utility gates.

### k-NN coverage-proxy decision

- All 24 tasks in `1111744` and selector `1111745` completed successfully. Selection root: `outputs/default_geocov_selection_1111744_1111745`. No arm passed every gate.
- `geo20_t0` improved consistency to Shape/Trend `+0.414319/+0.224751` with `6/6` wins for both and AUC `+0.004522`, but Beta remained `-0.313992`; lower penalties were worse on Beta. The proxy is directionally useful but insufficiently aligned with integrated manifold recall.

## Two-fold integrated train-coverage barrier

- New mechanism: shortlist eight sets using the locked moment + k-NN objective, then apply the same integrated precision/coverage geometry to two train-only reference folds with seeds derived from the generation seed and distinct from the evaluator's fixed seed. Selection uses the worst-fold recall as a barrier; no test/heldout rows or evaluator outputs are read.
- Four locked configurations vary only barrier penalty/tolerance: `int5_t0`, `int20_t0`, `int50_t0`, `int20_t005`. The proposal, priors, pair weights, coverage weights, shortlist size, and two-fold rule are fixed.
- Syntax smoke: `1111769`; 24-task screen at up to 12 GPUs: `1111770`; dependent fail-closed selector: `1111771`. Promotion requires zero two-fold train-coverage deficit on all six seeds plus the unchanged primary/utility gates.

### Integrated-coverage screen decision and confirmation

- All 24 tasks in `1111770` and selector `1111771` completed successfully. Selection root: `outputs/default_intcov_selection_1111770_1111771`.
- `int20_t005` passed all locked gates: Shape/Trend benefits `+0.409477/+0.186957` with `6/6` wins for both, Alpha `+0.512723`, Beta `-0.189073`, AUC `+0.004805`, and raw Shape/Trend `1.031849/1.469642`. Its two-fold train-coverage deficit was zero on every seed under the locked `.005` tolerance.
- The Beta margin is close to the `-0.20` floor, so this is promoted only to fresh confirmation, not declared final. No parameter or gate was changed.
- Fresh-confirmation syntax smoke: `1111808`; six new-pool GPU tasks: `1111809` (seeds `20261421..20261426`); dependent summary: `1111810`. Every seed regenerates the pool, tail voter, two-fold shortlist barrier, and final candidate from scratch.

### Confirmation integration correction

- Array `1111809` failed uniformly before candidate selection because the resampler validated the default integrated shortlist (`8`) against the control run's single draw even when integrated coverage was disabled. Pool generation completed; no evaluation result was produced, so this is a technical failure rather than evidence about the method.
- Validation was corrected to enforce `shortlist <= draws` only when the integrated barrier is enabled. The failed dependent summary `1111810` was cancelled. The same scripts, hyperparameters, and seeds were resubmitted unchanged as confirmation array `1111817` with summary `1111818`.

### Integrated-coverage fresh decision

- Corrected confirmation `1111817` and summary `1111818` completed successfully. Summary root: `outputs/default_intcov_confirmation_1111817_1111818`.
- The two-fold barrier fixed the isolated coverage failure: Beta benefit versus `G` became `+0.197890`, Alpha `+0.357292`, and AUC `+0.003436`. Shape remained robust at `+0.390959` with `6/6` wins and raw mean `1.011377`.
- The arm still failed the comprehensive rule because Trend benefit versus `G` was `+0.091889` with only `4/6` wins, and mean Trend versus the independently fixed random subset `R` was `-0.007453` with `2/6` wins. No gate was relaxed.

## Pair-robust selection under coverage barrier

- Next hypothesis: retain the successful two-fold coverage barrier unchanged while strengthening or broadening only the pair-discrepancy objective. Four locked arms are `pair3`, `pair6`, `pair3_m48`, and `pair3_t16`; they vary pair weight, total retained pair groups, or target-pair quota.
- Syntax smoke: `1111824`; 24-task reuse-pool screen: `1111825`; dependent selector: `1111826`. Promotion continues to require zero integrated-coverage deficit on all seeds and all original primary/utility gates.

### Pair-robust decision

- All 24 GPU tasks in `1111825` completed successfully. The first selector `1111826` failed because it used the preceding round's seed range; no experiment was rerun. The corrected CPU-only selector `1111850` completed, with selection root `outputs/default_pairrob_selection_1111825_1111850`.
- No arm passed. `pair3_t16` had the best Trend trade-off (Trend `+0.135604` vs `G`, `+0.036262` vs `R`) but still won only `4/6` and `3/6`; `pair3`, `pair6`, and `pair3_m48` did not improve the consistency failure. The Default pair-weight grid is stopped.

## Adult + Default graph-proximal transfer sweep

- Goal: test whether the graph-proximal quality-first output repair transfers to two additional classification datasets with dense local baselines, while preserving a direct ablation path from backbone (`B`) to dynamic graph generator (`G`), generic C3 selector (`old`), and graph-aware selector.
- Initial CPU-partition smoke `1114271` was cancelled while pending because the only requested CPU node was unavailable. Replacement smoke `1114273` uses the normal `prioritized` partition and gates all downstream work with `afterok`.
- Adult array `1114275` (`0-2%3`, seeds `20261501..20261503`) generates fresh 8,192-row pools with the previously selected categorical middle-injection operating point (`marginal_target`, strength `.25`, middle window `.20-.50`) over the same actuated factors `bscg_001,bscg_003`; it selects 4,096 rows. The frozen generic anchor is zero-shot `gt4`, built only from train marginals/pairs plus `G` and tail20 memberships.
- Default array `1114276` (`0-2%3`) reuses three clean fresh pools from confirmation `1111817` (seeds `20261421..20261423`), selects 8,192 rows, and keeps the original geometric + two-fold integrated train-coverage barrier active. No new Default diffusion generation is performed.
- Both datasets sweep the same four predeclared graph-proximal variants: `pqs0025_g05` (0.25% swaps, 1.5% quality budget, 5% required graph gain), `pqs005_g10` (0.5%, 3%, 10%), `pqs010_g10` (1%, 6%, 10%), and `pqs020_g25` (2%, 12%, 25%). Selection is quality-first after train-only graph/coverage feasibility.
- Every task evaluates paired `B/G/R/old` plus all four variants, writes source hashes, GPU metadata, selected indices, full train-only selection traces, graph audits, and common fidelity/utility metrics.
- Dependent summary `1114277` runs only after both arrays succeed and compares Adult against DynamicGraph, ForestDiffusion, TabSyn, CTGAN, TVAE, and SAGE tables; Default is compared against the local baseline suite plus TabSyn and ForestDiffusion. Winner selection remains fail-closed on graph feasibility, gain over `G`, and the frozen-selector floor.

### Transfer decision and deficit-activated confirmation

- Adult `1114275`, Default `1114276`, and summary `1114277` completed successfully. Summary root: `outputs/graph_proximal_transfer_selection_1114275_1114276_1114277`.
- Adult `pqs0025_g05` achieved Shape/Trend `0.805368/1.919238`, gains versus `G` of `+0.465029/+0.585671`, gains versus frozen `gt4` of `+0.011023/+0.036369`, and `4/5` local SOTA metric wins. It improved both primary metrics over `G` on all three seeds and over `gt4` on two of three seeds, but the original train-only graph gate was feasible on only `2/3` seeds.
- The Adult failure is a diagnosed low-deficit boundary case, not an unconstrained quality failure. On seed `20261503`, the frozen anchor graph relation loss was already `0.000461`; forcing ten swaps plus a further relative 5% graph reduction made all 256 candidates worse. On the other two seeds, 210/256 and 255/256 candidates were feasible and improved graph relation by `0.000826` and `0.001830`.
- Default `pqs0025_g05` passed all locked gates on `3/3` seeds. Mean Shape/Trend were `1.030677/1.554413`; gains versus `G` were `+0.432777/+0.180281` with `3/3` wins for both, and gains versus `int20_t005` were `+0.034284/+0.023674`, also `3/3` for both. Graph relation improved by `+0.002772`; local SOTA wins were `2/5`. This is a coherent positive transfer, though not yet a five-metric SOTA claim.
- Method correction: graph-proximal repair is now deficit-activated. If the frozen anchor's train-only graph relation loss is at or below `.001`, the graph gate preserves the anchor (zero swaps) rather than demanding an ill-conditioned relative improvement. Above the threshold, the locked `pqs0025_g05` repair is unchanged. Selection metadata records whether activation was skipped, the realized swap count, and the reason.
- Data smoke `1114358` validates syntax plus the expected no-op on the diagnosed Adult seed. Fresh/held-out confirmation array `1114359` (`0-5%6`) is dependency-gated on the smoke: Adult uses fresh seeds `20261511..20261513`; Default uses the three previously untouched `1111817` pools (`20261424..20261426`). Dependent summary: `1114360`.

### Deficit-gate fresh/held-out decision

- Smoke `1114358`, six-task confirmation `1114359`, and summary `1114360` all completed successfully. Summary root: `outputs/graph_deficit_gate_confirmation_1114359_1114360`.
- Adult is a comprehensive fresh-seed result. The deficit gate was feasible on `3/3` seeds and correctly no-op on one already-consistent anchor. Mean Shape/Trend were `0.860949/2.017464`; gains versus `G` were `+0.371348/+0.521812` with `3/3` wins for both, gains versus frozen `gt4` were `+0.008133/+0.024268`, and graph relation improved by `+0.000300` without coverage loss. Alpha/Beta/AUC were `99.068206/48.155382/.897593`; the method set `4/5` local baseline records. The locked summary decision is `comprehensive`.
- Default did not confirm as a graph-repair improvement. It remained graph-feasible on `3/3` held-out pools and improved graph relation by `+0.002588`, but Trend beat `G` on only `1/3` seeds and the candidate was slightly worse than frozen `int20_t005` on mean Shape/Trend (`-0.007101/-0.000045`), AUC (`-0.005883`), and the locked quality floor. The locked summary decision is `reject`. The earlier first-three-seed result is retained as screening evidence only.
- Final interpretation: deficit activation fixes the low-loss boundary condition and establishes the proposed graph-shared proximal repair on Adult. Shoppers remains a coherent transfer result. Default, Magic, and Diabetes are negative/boundary evidence for universal graph-repair claims; their stronger frozen-C3 results remain valid independent ablations.

## News zero-shot particle transfer

- To test whether the method offers a cross-dataset claim despite the remaining Default comprehensive gate, the locked `gt4` and `int20_t005` mechanisms are transferred to News without retuning. Each seed generates one 16,384-row locked-News-`G` pool, then constructs both 8,192-row selectors and evaluates `B/G/R/gt4/int20_t005` together.
- News transfer seeds: `20261431,20261432`; GPU array: `1111854` (`0-1%2`); dependent CPU summary: `1111855`; syntax smoke: `1111853`.
- Screen gates require Shape/Trend wins on both seeds versus `G` and `R`, positive means versus `B/G/R`, bounded Alpha/Beta/R2/NRMSE downside, and a satisfied two-fold train barrier for `int20_t005`.

### News zero-shot transfer decision and primary confirmation

- Both tasks in `1111854` and summary `1111855` completed successfully. Summary root: `outputs/news_particle_transfer_summary_1111854_1111855`. Neither arm passed the comprehensive two-seed gate.
- `int20_t005` showed large zero-shot benefits versus News `G`: Shape `+0.504243` and Trend `+0.205119` (`2/2` each), R2 `+0.456485`, and NRMSE benefit `+0.141512`; it lost Alpha `-0.318632` and Trend won only `1/2` versus random subset `R`.
- `gt4` had Shape/Trend `+0.526138/+0.333015` but worse Alpha/Beta (`-0.336128/-0.245768`). `int20_t005` is retained as the better utility-aware transfer arm.
- To separate a valid main-metric claim from a comprehensive claim, `int20_t005` is locked for six fresh News seeds. Primary confirmation requires positive Shape/Trend means versus `B/G/R`, at least `5/6` wins versus `G/R`, and positive bootstrap intervals. Comprehensive confirmation additionally retains the original Alpha/Beta/R2/NRMSE floors.
- Syntax smoke: `1111857`; six-GPU confirmation array: `1111858` (seeds `20261441..20261446`); dependent dual-label summary: `1111859`.

### News six-seed decision

- All six GPU tasks in `1111858` completed successfully. The original summary `1111859` treated one missed internal barrier as malformed input; the corrected CPU-only summary `1111866` recorded it as a confirmation rejection without rerunning experiments. Summary root: `outputs/news_intcov_confirmation_1111858_1111866`.
- Shape transfer is robust: benefit versus News `G` `+0.463686`, wins `6/6`, bootstrap 95% interval `[+0.407310,+0.514096]`. R2 and NRMSE benefits were `+0.248646/+0.086320`.
- Trend did not transfer robustly: mean benefit versus `G` `-0.113013`, wins `2/6`, interval `[-0.458594,+0.253668]`; versus `R`, mean was `+0.131500` but only `4/6` wins. One seed also missed the locked two-fold train-coverage barrier, and Alpha was marginally below its floor (`-0.104054 < -0.10`).
- Decision: neither primary-metric nor comprehensive News confirmation passed. Retain the method as a robust Shape-specialist across Default and News, not as a cross-dataset Shape+Trend SOTA claim. Stop the terminal-selection grid; the next scientific branch should move the coverage/pair barrier into timestep-aware particle interaction or explicitly optimize a Pareto frontier rather than another scalar-weight sweep.

## Baseline-rich cross-dataset particle transfer

- Dataset inventory job `1111881` completed on the CPU partition. Among datasets not used for the Default/News confirmations, Shoppers, Magic, and Diabetes have the densest local result coverage and complete TabDiff checkpoint/evaluator/graph assets. The inventory artifact is `outputs/multidataset_inventory_1111881`.
- GPU screen `1111895` runs nine paired tasks at up to nine-way concurrency: three datasets (`shoppers`, `magic`, `diabetes`) by three base generators (`plain`, bounded middle projection, and a projection + target-conditional copula + categorical hybrid). Identical base seeds within each dataset make the unconditional samples directly auditable across the three generator arms.
- Each task generates one 8,192-row pool and applies five train-only 4,096-row selectors without regenerating diffusion samples: locked `gt4`, locked `int20_t005`, marginal-only stability (`marg4`), target/pair-heavy stability (`trend4`), and a stricter integrated-coverage barrier (`int50_t0`). Evaluation also includes paired `B/G` and a random subset `R`.
- The 4,096-row output budget matches the existing local official-generator protocol. Any SOTA comparison will use the same repository split and evaluator against locally generated TabSyn, ForestDiffusion, CTGAN, TVAE, and TabDDPM rows where available; paper-reported numbers from incompatible protocols remain excluded.
- No heldout/test rows enter generation or subset selection. Per-task provenance records the checkpoint, graph, code hashes, GPU, seed, base generator, and selector metadata. Output pattern: `outputs/multidataset_particle_<dataset>_<base>_1111895_<task>`.
- Downstream pipeline syntax smoke `1111940` completed successfully. CPU selector `1111942` is queued `afterok:1111895`; it ranks one base+selector per dataset using paired Shape/Trend improvement, utility floors, local official-baseline coverage, then a predeclared normalized score.
- Fresh three-seed confirmation array `1111943` (`0-8%9`) is queued `afterok:1111942`, followed by CPU summary `1111944`. Confirmation regenerates every diffusion pool and train-only selector at seeds `20261461..20261469`; it does not reuse screening particles.

### Cross-dataset confirmation decision

- Screen `1111895`, selector `1111942`, fresh confirmation `1111943`, and summary `1111944` all completed successfully.
- Diabetes `hybrid+int20_t005`: Shape/Trend `1.353019/8.165711`, gains versus `G` `+0.319212/+1.137281` with `3/3` wins for both; Alpha/Beta/AUC `93.636610/51.369358/.651681`; `3/5` local SOTA metric records. Decision: confirmed primary-metric result.
- Magic `proj+gt4`: Shape/Trend `1.009621/1.060297`, gains versus `G` `+0.318999/+0.887710` with `3/3` wins for both; Alpha/Beta/AUC `98.609484/49.172635/.925748`; `3/5` local SOTA metric records. It also beat random `R` on Shape in `3/3` seeds and Trend in `2/3`. Decision: comprehensive confirmation.
- Shoppers `proj+gt4`: Shape/Trend `1.097399/2.096839`, gains versus `G` `+0.543518/+0.107746`, Shape wins `3/3` and Trend wins `2/3`; Alpha/Beta/AUC `98.158188/48.007812/.904839`; `2/5` local SOTA metric records. Decision: confirmed primary-metric result.
- The strongest generator+selector pairs are therefore frozen as the baseline for graph-consistent output selection. No selector hyperparameter is retroactively changed.

## Graph-consistent output selection

- Scientific goal: close the semantic gap between dynamic graph guidance and terminal particle selection. The exact factor edges actuated during sampling define a three-state statistic per row: not applicable, applicable-and-satisfied, and applicable-and-violated. Selection preserves the train distribution of these states, penalizes factor-support undercoverage, and optionally penalizes excess hard-factor violation.
- The frozen generic C3 selectors remain intact and independently evaluable (`old`). New ablations are `graph_only` (no generic C3 objective) and five additive `gc_*` variants. Thus the method can report dynamic generator only (`G`), frozen generic selector (`old`), graph-only selection, and combined dynamic+generic+graph selection without weakening the old arm.
- Only active graph factors are shared between sampling and selection: Shoppers `r1_014,seed_003,r2_006,r1_015,r1_005,r1_021,r1_013`; Magic `bscg_000..004`; Diabetes `bscg_011`. Edge weights combine usage role, confidence, bootstrap stability, and a bounded log selection-score multiplier.
- Graph audit is computed by one shared function for both `old` and every new subset, preventing the method from claiming graph preservation using an incomparable internal objective. Per-edge support, violation rate, relation L1, and aggregate coverage/hard excess are persisted.
- Compile/import smoke `1112399`; functional data smoke `1112409` (dependent on compile smoke). The nine-task reuse-pool screen `1112400` depends on the functional smoke and tests `graph_only`, `gc_lite`, `gc_bal`, `gc_strong`, `gc_cov`, and `gc_hard` on the nine fresh pools from `1111943`. It evaluates `B/G/R/old` plus every graph arm in one evaluator call per seed.
- CPU selector/summary `1112401` is dependent on screen `1112400`. The predeclared ranking first requires graph-relation improvement with non-worse coverage/hard excess, then a utility floor versus frozen C3, joint Shape/Trend improvement versus C3, retained gain versus `G`, local SOTA record count, and finally a scalar tie-break score.
- Confirmation-pipeline syntax smoke `1112406`. Fresh-pool three-dataset confirmation `1112407` (`0-8%9`, seeds `20261481..20261489`) depends on both the screen selection and confirmation smoke; CPU summary `1112408` depends on confirmation. Each confirmation task regenerates an 8,192-row dynamic pool, reconstructs both frozen C3 and the selected graph-consistent arm, audits them on identical active factors, and evaluates `B/G/R/old/graph` together.

### Scalar graph-selection decision

- The original CPU dependency chain was replaced only for queue latency, without changing experiment inputs: accelerated compile smoke `1112506`, confirmation syntax smoke `1112507`, functional data smoke `1112508`, screen `1112400`, GPU summary `1112692`, fresh confirmation array `1112695`, and summary `1112696` all completed with exit code zero. Superseded pending CPU jobs `1112401`, `1112407`, and `1112408` were cancelled before execution.
- Functional smoke improved graph relation L1 from `.019140` to `.014700`, graph coverage gap from `.000562` to `.000517`, and hard excess from `.008126` to `.007500`, establishing that the shared graph audit and selector operate in the intended direction.
- Frozen-pool winners were Diabetes `gc_lite`, Magic `gc_bal`, and Shoppers `gc_cov`. Each improved the main graph relation metric on all three screen seeds, but none passed the unchanged frozen-C3 utility floor. In particular, Magic gained Shape/Trend `+.048996/+.105091` versus old C3 but lost enough secondary utility to miss the floor; Shoppers traded Shape loss `-.110075` for Trend gain `+.025439`.
- Fresh confirmation root: `outputs/graph_consistent_confirmation_1112695_1112696`. Diabetes retained Shape/Trend gains versus old C3 of `+.032037/+.262422`, while Magic changed them by `-.031124/-.370569` and Shoppers by `-.177187/-.434525`. Magic and Shoppers are rejected; the scalar graph penalty is not promoted as a cross-dataset method.
- Scientific diagnosis: adding graph discrepancy as another scalar objective exposes a Pareto conflict with the generic train-distribution objective. The next branch therefore treats frozen C3 as an explicit train-only quality anchor and graph improvement as a constraint, rather than tuning another arbitrary additive weight.

## Graph-constrained Pareto anchored selection

- New mechanism: sample candidate subsets from the same dynamic-pool/C3 proposal, require a declared relative graph-relation improvement plus non-worse graph coverage and hard-factor excess, and bound degradation of the frozen generic C3 train-only objective. Only candidates satisfying all train-only constraints are eligible; selection then minimizes generic quality discrepancy and graph relation lexicographically. Evaluator metrics remain completely outside selection.
- The old C3 subset is a frozen anchor and remains an independent ablation. This gives a coherent chain: the graph defines row-level diffusion intervention, the same active factors define set-level admissibility, and the generic C3 objective prevents graph optimization from manufacturing an easy result by sacrificing distribution quality.
- Four configurations were fixed before evaluator access: `pa_g0_q10`, `pa_g10_q15`, `pa_g25_q20`, and `pa_g50_q30`. They require graph-relation gains of `0/10/25/50%` with generic train-quality tolerances of `10/15/20/30%`; only the latter three change graph-raking strength. The tolerances were calibrated from the selector's train-only objective after a 4-draw smoke showed the earlier `5%` bound infeasible; no heldout/evaluator metric was used.
- Execution smoke `1114179` completed successfully. The nine frozen-pool GPU tasks are submitted as `1114186` (`0-8%3`), with dependent GPU summary/selector `1114187`. Each task evaluates all four locked variants against identical `B/G/R/old` references and saves feasibility counts, generic-objective gaps, graph audits, public metrics, provenance, and row indices.
- Fresh confirmation is prepared to reuse the already generated pools from `1112695`, avoiding another diffusion run. Confirmation syntax/compile smoke is `1114189`. After `1114187` chooses one fail-closed winner per dataset, exactly one nine-task selector/audit/evaluation array will run on seeds `20261481..20261489`, followed by one summary. No server-side agent or login-node compute is used.

### Early complete-dataset promotion: Magic

- All three Magic tasks in screen `1114186` completed before the integrated-coverage Diabetes tasks. The unchanged summarizer was run on the complete Shoppers+Magic slice as job `1114223`; this is a full three-seed dataset decision, not an early-seed decision.
- Magic `pa_g25_q20` passed every locked screen gate: all `3/3` train constraints feasible, mean graph-relation gain versus frozen C3 `+.004907`, Shape/Trend gains `+.019458/+.082075`, Alpha/Beta/AUC changes `+.276206/-.091688/-.002787`, dynamic-generator gain retained, and `3/5` local SOTA metric records.
- Shoppers `pa_g10_q15` was rejected by the unchanged quality floor (Shape `-.054354`, AUC `-.011007` versus C3) despite a strong Trend gain; it is not promoted.
- Exactly the promoted Magic configuration is submitted on the three fresh pools from `1112695` as array `1114225` (`3-5%3`), with summary `1114226`. This reuses generated particles but recomputes anchored selection, graph audit, and all evaluator metrics independently for seeds `20261484..20261486`.
- Fresh Magic confirmation `1114225/1114226` completed and rejected the full-redraw Pareto arm. It remained feasible `3/3` and improved graph relation by `+.002760`, but changed Shape/Trend versus C3 by `-.048167/+.028892` and Trend versus dynamic `G` by `-.011890`; only `2/5` local SOTA records remained. This isolates redraw distance, rather than lack of graph correction, as the generalization failure.

## Graph-proximal sparse membership repair

- Parallel follow-up hypothesis: the quality instability above may come from redrawing an entire 4,096-row subset. Start from frozen C3 and make only sparse membership swaps, using first-order residuals of the active graph-factor state distributions to choose selected rows to remove and unselected rows to add. This is a proximal set update tied to the same factor graph used in dynamic guidance, not another marginal/pair scalar selector.
- The unchanged constraints require a positive graph-relation gain, non-worse graph coverage and hard-factor excess, and bounded degradation of the generic train-only C3 objective. Candidate selection never reads evaluator outputs. The old C3 subset remains exactly available as the zero-swap ablation.
- Execution smoke `1114197` completed successfully. A 1%/41-row repair reduced Shoppers graph relation from `.019140` to `.014972`; eight smoke draws placed the best generic objective `4.72%` above the anchor. This train-only feasibility observation was used before the screen to lock four budgets: `ps005_q03`, `ps010_q06`, `ps025_q12`, and `ps050_q20`, meaning `0.5/1/2.5/5%` swapped rows with `3/6/12/20%` train-objective tolerances. No evaluator metric was inspected for this calibration.
- Nine-task frozen-pool screen `1114201` (`0-8%3`) and dependent summary `1114202` are submitted. Every task evaluates all four repair budgets plus identical `B/G/R/old` references, records exact changed membership, feasibility counts, factor-level audit, generic objective, evaluator metrics, and provenance.
- Early graph-first results showed the intended graph/Trend correction but a small, systematic Shape cost on Shoppers. The mechanism is therefore ablated with a second, predeclared lexicographic ordering: first minimize generic C3 train quality among candidates that already achieve at least `10%` or `25%` graph-relation gain, then minimize graph loss. It uses only `0.5%` or `1%` swaps and keeps the same feasibility constraints.
- Quality-first smoke `1114214` completed successfully: 20 swaps increased the generic objective by only `0.10%` while reducing graph relation by about `24.5%`. The locked four-arm quality-first screen is `1114215` (`pqs005_g10`, `pqs005_g25`, `pqs010_g10`, `pqs010_g25`), with dependent summary `1114216`.
- Complete three-seed Shoppers+Magic summary `1114235` selected `pqs005_g10` for both datasets. Magic passed with Shape/Trend `+.063301/+.004510`, graph `+.003291`, Alpha/Beta/AUC `+.053711/-.108507/+.001075`; Shoppers passed with Shape/Trend `+.000664/+.003940`, graph `+.002239`, Alpha/Beta/AUC `+.084093/+.011936/-.001981`. Both were feasible `3/3` and passed constraint, utility-floor, primary, and dynamic gates.
- Confirmation-pipeline smoke `1114238` completed. Six fresh-pool tasks are submitted as `1114240` (Shoppers+Magic, seeds `20261481..20261486`), with dependent summary `1114241`. The method is locked to exactly 20 membership swaps, at least 10% graph-relation gain, a 3% generic train-objective tolerance, and quality-first lexicographic choice.

### Quality-first fresh decision

- Confirmation `1114240/1114241` completed successfully; every selector was train-feasible (`3/3` per dataset), all graph audits improved relation, and no diffusion pool was regenerated.
- Shoppers is confirmed `coherent`: graph relation/coverage gains versus C3 `+.002233/+.000619`; Shape/Trend gains versus dynamic `G` `+.389162/+.234395`; raw Shape/Trend `1.102046/1.803424`; Alpha/Beta/AUC `98.909617/48.103841/.902240`; `3/5` local SOTA metric records. The sparse repair changes Shape/Trend versus C3 by only `-.024145/-.030366`, inside all frozen utility floors.
- Magic improves C3 on Shape/Trend by `+.008858/+.007241` and graph relation by `+.001749`, but is rejected because Trend remains `-.033541` versus dynamic `G`; this failure is recorded rather than relabeled. Raw Magic Shape/Trend are `.956562/2.334173`, with `2/5` local SOTA records.
- Current evidence therefore supports a dataset-level coherent claim for Shoppers and a general mechanism/ablation claim, not yet a comprehensive cross-dataset claim. The frozen old C3 arm remains independently strong, while sparse graph repair provides graph-semantic coupling at bounded quality cost.
