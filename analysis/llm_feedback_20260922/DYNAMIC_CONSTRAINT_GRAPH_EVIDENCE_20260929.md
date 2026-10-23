# Dynamic constraint graph: formulation and current evidence

## 1. Correct formulation

The proposed direction is not simply “activate an edge whenever a column has high uncertainty.” The intended architecture is:

1. An LLM compiles schema, metadata, domain description and a user request into a typed prior graph `G_prior`.
2. A deterministic type checker and verifier reject invalid nodes/edges and calibrate them on training data.
3. At selected reverse steps, the system reads the current clean estimate `x_hat_0(x_t)` and constructs a row-specific active graph:

   `G_(i,t) = Select(G_prior, request_i, x_hat_0(i,t), trajectory_i, uncertainty_i)`.

4. The active graph determines which columns receive guidance, guidance strength, whether to defer, abstain or rollback.
5. The LLM is called again only for high-uncertainty, conflicting or non-executable semantic cases.

The desired semantic activation score is conceptually:

`activation(i,e,t) = readability(i,t) * applicability(i,e,t) * violation_probability(i,e,t) * (1 - exception_probability(i,e,t))`.

Therefore:

- unreadable/highly noisy preview: defer rather than automatically actuate;
- not applicable or already satisfied: deactivate or use near-zero guidance;
- confidently applicable and violated: activate;
- possible valid exception or graph conflict: abstain or request LLM feedback.

Nodes should represent predicates, columns/values, applicability, exceptions and actions. Typed edges should distinguish `requires`, `implies`, `conflicts`, `exception_to`, `depends_on` and `actuates`.

## 2. What has actually been implemented

The current experiment is a mechanism probe, not the full LLM system.

- The graph prior is constructed from normalized mutual information on the training split, not by an LLM.
- Condition columns are always active.
- Candidate dependency columns are scored independently for each synthetic row and reverse step.
- For a numeric candidate, response strength is the absolute conditional denoiser delta.
- For a categorical candidate, response strength is the RMS conditional logit delta; normalized entropy increases its uncertainty factor.
- Each score is multiplied by the train-only dependency prior, and the top `k` candidate columns are activated.
- If a column is active, its complete conditional guidance delta is used. Otherwise it receives the unconditional prediction.

The implemented heuristic is approximately:

`score(i,j,t) = conditional_response(i,j,t) * dependency_prior(j) * categorical_uncertainty(i,j,t)`.

This heuristic intentionally asks whether a row/timestep-dependent topology can matter. It does not yet detect applicability, satisfaction, violation or valid exceptions. Numeric uncertainty is not explicitly modelled.

## 3. Experimental protocol

Dynamic graph is paired against a fixed top-`k` dependency graph using the same:

- TabDiff backbone and condition adapter;
- dataset, condition and held-out reference cohort;
- adapter seed and sampling seed;
- reverse-step schedule and guidance scale;
- number of active dependency columns;
- 2,048 samples per condition and 30 reverse steps;
- two denoiser evaluations per trajectory.

Automatic graph candidates exclude supervised evaluation targets and protected/sensitive columns unless the user condition explicitly mentions them.

The full submitted grid contains 216 runs over Default and Shoppers, numeric and mixed condition catalogs, three seeds, three schedules, active budgets `{1,2,4}` and candidate pools `{4,8}`. Job `1048642[0-215]` is still running; jobs `1048643–1048645` will aggregate it.

## 4. Completed results so far

### 4.1 Mixed-type smoke test

Default, mixed conditions, mid-pulse guidance, active budget 2, candidate pool 8, one seed:

| Rarity | Delta success | Delta denoiser calls/valid | Delta KS | Delta TV |
| --- | ---: | ---: | ---: | ---: |
| q750 | +0.0493 | -0.87 | +0.0045 | -0.0019 |
| q900 | +0.0697 | -1.95 | +0.0104 | +0.0042 |
| q950 | +0.0907 | -3.21 | +0.0240 | -0.0045 |
| q975 | +0.1175 | -4.41 | +0.0290 | +0.0025 |
| Macro mean | +0.0818 | -2.61 | +0.0169 | +0.0001 |

Positive success delta favors dynamic graph. Negative KS/TV favors dynamic graph. The result shows substantially higher condition satisfaction and lower expected sampling cost, almost unchanged categorical TV, but worse numeric KS. It is a trade-off rather than Pareto dominance and currently has only one mixed-condition seed.

### 4.2 Completed numeric configurations

Default, constant guidance. Rows with three seeds are complete; the final `4-of-8` row is currently based on one seed.

| Active/Pool | Seeds | Paired cases | Delta success | Delta calls/valid | Delta KS | Delta TV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/4 | 3 | 54 | -0.0004 | -0.024 | -0.0044 | -0.0024 |
| 1/8 | 3 | 54 | +0.0007 | +0.008 | -0.0007 | -0.0020 |
| 2/4 | 3 | 54 | +0.0003 | -0.015 | -0.0042 | -0.0009 |
| 2/8 | 3 | 54 | +0.0032 | -0.006 | -0.0011 | -0.0027 |
| 4/4 | 3 | 54 | +0.0000 | +0.000 | +0.0000 | +0.0000 |
| 4/8 | 1 | 18 | +0.0101 | -0.129 | -0.0018 | -0.0029 |

`4-of-4` is the expected sanity control: every candidate edge is selected, so dynamic and fixed graphs are identical. Numeric results with smaller budgets are mostly neutral in success and slightly favorable in fidelity. The clearest three-seed adaptive result, `2-of-8`, improves success by only 0.0032, so it is not practically compelling by itself.

At the rarest numeric split q990:

- `1-of-4`: success `+0.0030`, KS `-0.0079`, TV `-0.0039`;
- `2-of-4`: success `+0.0049`, KS `-0.0081`, TV `-0.0021`;
- `2-of-8`: success `+0.0024`, KS `-0.0033`, TV `-0.0056`.

These signs are favorable but the effects are small.

### 4.3 Evidence that the graph is dynamic

In the mixed smoke run, mean edge-selection change between consecutive reverse steps is approximately 0.118. Selected edges and selection frequencies differ across rows and early/middle/late phases. This establishes that the implementation changes the active topology rather than merely renaming a fixed graph.

## 5. Supporting evidence from earlier experiments

The following results motivate the design but do not prove the complete method:

- No universal temporal guidance schedule wins across datasets and conditions.
- Target-only and dependency-scoped guidance improve fidelity but reduce condition success, showing that actuation scope matters.
- The cross-dataset rule audit finds exact one-way implications, near-hard asymmetric relations, probabilistic associations and valid subgroup exceptions. A single static hard graph is therefore unsafe.
- Direct per-record LLM judgement is weaker and much more expensive than compile-once execution:
  - GPT-4.1-mini direct: accuracy 0.864, request-swap accuracy 0.867, about 37.5k tokens;
  - GPT-5.4-mini direct: accuracy 0.650, request-swap accuracy 0.367, about 37.7k tokens;
  - LLM to typed DSL plus deterministic execution: accuracy 1.000, request-swap accuracy 1.000, about 1.85k tokens.

These results support an LLM-compiled prior graph with deterministic routine updates, not an LLM call at every reverse step.

## 6. What is not yet demonstrated

Current evidence does not show that:

- an LLM can construct a sufficiently accurate graph from schema/metadata;
- an LLM can reliably add, remove or reweight graph edges during diffusion;
- uncertainty is a calibrated signal of constraint violation;
- dynamic graph preserves valid exceptions or rare modes;
- the method beats CTDF-style projection, Harpoon-style guidance, best-of-N or post-generation repair on the same protocol;
- gains generalize across datasets, schedules and seeds;
- the current method is Pareto-superior in validity, fidelity and cost.

## 7. Current conclusion and permissible claim

The complete LLM-controlled dynamic semantic graph is not yet empirically validated. The defensible current conclusion is:

> A row- and timestep-dependent actuation graph is technically viable and genuinely changes topology. Preliminary mixed-type evidence shows higher condition satisfaction and lower sampling cost than a fixed graph, at a small numeric-fidelity cost. Numeric-only evidence is mostly neutral. The LLM graph constructor/controller remains untested.

The next decisive experiment should use an oracle four-state graph (`violated`, `satisfied`, `not applicable`, `valid exception`). If the oracle graph cannot improve the validity-fidelity frontier over fixed projection and repair, adding an LLM controller is not justified. If the oracle wins, the LLM component can be evaluated as an approximation to oracle graph selection, with explicit calibration, abstention and cost metrics.

## 8. Result locations

- Current full partial table: `outputs/dynamic_graph_partial_full_20260929_065108/RESULTS.md`
- Current paired rows: `outputs/dynamic_graph_partial_full_20260929_065108/paired_rows.csv`
- Current grouped summary and confidence intervals: `outputs/dynamic_graph_partial_full_20260929_065108/summary.csv`
- Final paired aggregate after grid completion: `outputs/dynamic_graph_comparison_1048645/`
- Main implementation: `scripts/temporal_contract_guidance.py`
- Comparison implementation: `scripts/dynamic_graph_comparison.py`

## 9. Metric definitions and absolute results

### 9.1 Metrics

**Single-trajectory success** is the fraction of all generated rows that satisfy every predicate in the requested condition before rejection or repair. For example, 0.35 means that approximately 35 of 100 trajectories directly produce a valid requested row. Higher is better.

**Normalized denoiser calls per valid row** is currently logged as `2 / success`, because guided sampling evaluates one conditional and one unconditional prediction per reverse step. All experiments use 30 reverse steps. Therefore actual backbone forward evaluations per valid row are `30 * logged_calls`. Lower is better. This is an expected cost under independent resampling, not measured wall-clock latency.

**Numeric KS** is the mean two-sample Kolmogorov-Smirnov statistic over numeric columns. It compares accepted synthetic rows with held-out real rows satisfying the same condition. KS is the largest difference between two empirical CDFs, ranges from 0 to 1, and lower is better.

**Categorical TV** is the mean total-variation distance over categorical columns between accepted synthetic rows and the corresponding held-out real cohort. It ranges from 0 to 1, and lower is better.

Success is evaluated on all generated trajectories. KS and TV are evaluated after retaining generated rows that satisfy the request; they measure conditional fidelity rather than unconditional table fidelity. Percentage changes in KS/TV should be interpreted cautiously when the base value is already small, so both absolute and relative changes are reported.

### 9.2 Mixed conditions: absolute values

| Rarity | Fixed success | Dynamic success | Absolute gain | Relative gain | Fixed calls | Dynamic calls | Fixed KS | Dynamic KS | Fixed TV | Dynamic TV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q750 | 0.3429 | 0.3923 | +0.0493 | +14.4% | 6.79 | 5.92 | 0.1042 | 0.1087 | 0.0514 | 0.0495 |
| q900 | 0.2878 | 0.3574 | +0.0697 | +24.2% | 8.46 | 6.51 | 0.1309 | 0.1413 | 0.0659 | 0.0701 |
| q950 | 0.2339 | 0.3245 | +0.0907 | +38.8% | 10.10 | 6.90 | 0.1456 | 0.1696 | 0.0907 | 0.0862 |
| q975 | 0.1947 | 0.3122 | +0.1175 | +60.4% | 11.12 | 6.71 | 0.2176 | 0.2466 | 0.1061 | 0.1086 |
| Macro mean | 0.2648 | 0.3466 | +0.0818 | +30.9% | 9.12 | 6.51 | 0.1496 | 0.1665 | 0.0785 | 0.0786 |

At the macro level, dynamic graph raises direct validity from 26.5% to 34.7%, a gain of 8.2 percentage points or 30.9% relative. Normalized calls fall by 28.6%; with 30 steps, actual expected backbone evaluations fall from about 273.5 to 195.3 per valid row. Numeric KS increases by 0.0169, or 11.3% relative, while categorical TV is effectively unchanged.

### 9.3 Numeric conditions: absolute macro values

| Active/Pool | Seeds | Fixed success | Dynamic success | Relative gain | Fixed calls | Dynamic calls | Fixed KS | Dynamic KS | Fixed TV | Dynamic TV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/4 | 3 | 0.4642 | 0.4638 | -0.08% | 5.541 | 5.516 | 0.1435 | 0.1391 | 0.0739 | 0.0715 |
| 1/8 | 3 | 0.4642 | 0.4650 | +0.16% | 5.541 | 5.549 | 0.1435 | 0.1427 | 0.0739 | 0.0719 |
| 2/4 | 3 | 0.4782 | 0.4785 | +0.05% | 5.166 | 5.151 | 0.1441 | 0.1399 | 0.0771 | 0.0762 |
| 2/8 | 3 | 0.4782 | 0.4814 | +0.67% | 5.166 | 5.160 | 0.1441 | 0.1430 | 0.0771 | 0.0744 |
| 4/4 | 3 | 0.4899 | 0.4899 | 0.00% | 4.941 | 4.941 | 0.1440 | 0.1440 | 0.0761 | 0.0761 |
| 4/8 | 1 | 0.4903 | 0.5004 | +2.05% | 4.933 | 4.805 | 0.1433 | 0.1415 | 0.0772 | 0.0744 |

For the completed three-seed `2-of-8` setting, direct validity changes only from 47.82% to 48.14%. KS improves by 0.0011 and TV by 0.0027. Thus numeric-only benefits are small. The one-seed `4-of-8` setting is more favorable but is not yet replicated.
