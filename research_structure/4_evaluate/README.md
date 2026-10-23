# Evaluation

Question: **Did graph validity improve without sacrificing realism, coverage, utility, privacy, or compute fairness?**

## Primary metrics

- Shape error/score: univariate fidelity; direction is protocol-specific.
- Trend error/score: cross-column fidelity; direction is protocol-specific.
- Alpha: precision-like realism.
- Beta: recall-like mode coverage.
- TSTR: downstream utility.
- Graph relation loss / coverage gap / hard excess: internal diagnostic, never the only success criterion.
- DCR/privacy and duplicate/novelty checks where available.

## Main implementation families

- `evaluate_shoppers_baseline_suite.py`
- `audit_graph_consistency.py`
- `paired_claim_audit.py`
- `posthoc_independent_audit.py`
- `rare_support_statistical_audit.py`
- `subgroup_selector_stat_audit.py`
- `aggregate_paper_tables_20261011.py`

Never use held-out/test evaluator metrics as selector inputs. Keep score-vs-error directions explicit, especially for News versus Adult/Shoppers/Magic/Default.
