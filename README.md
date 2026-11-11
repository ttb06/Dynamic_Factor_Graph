# DyGRAFT

Implementation of DyGRAFT: record-level diffusion guidance and set-level tabular control through a shared verified factor graph. The code uses a frozen TabDiff backbone, followed by distribution-aware anchoring, support restoration, and graph-guided refinement.

## Setup

Use Python 3.11 and a CUDA-capable PyTorch installation for sampling and training. Selection and evaluation run on CPU.

```bash
python -m pip install -r requirements.txt
python -m dygraft --help
```

`requirements.txt` records the versions used for the final runs. Install the PyTorch wheel appropriate for your CUDA installation. Eight CPU threads and 32 GB host memory are a practical starting point; run all compared variants with the same thread count and GPU model.

## Data and checkpoints

The [submission release](https://github.com/ttb06/Dynamic_Factor_Graph/releases/tag/submission-20261112) provides `benchmark_splits.tar.gz` and `checkpoints.tar.gz`. Extract both into the repository root to use the frozen splits and backbones from the experiments. Checkpoints are stored as `checkpoints/<dataset>/model.pt`; their readable model configurations are in `configs/models/`.

Alternatively, prepare public inputs and retrain:

```bash
python -m dygraft prepare --dataset adult
python -m dygraft train --dataset adult
```

The preparation command supports `adult`, `bank`, `churn`, `default`, `insurance`, `magic`, `shoppers`, or `all`. Adult uses its supplied training/test files; the other datasets use the original deterministic 90/10 split. Numerical transformations are fitted on training data. Insurance uses a numerical regression target.

Raw sources: [UCI](https://archive.ics.uci.edu/), the [CDTD benchmark inputs](https://github.com/muellermarkus/cdtd/tree/main/data/raw_data) for Bank and Churn, and the [LCATS Insurance input](https://github.com/SJTU-DMTai/LCATS/tree/main/data/insurance). Dataset rights remain with their original providers. Use the frozen split asset for exact row ordering and preprocessing compatibility.

The Adult, Default, Magic, and Shoppers backbones use the included training loop; Bank, Churn, and Insurance use the included upstream TabDiff trainer. `configs/datasets/` records each training configuration. Training and generation discretizations can differ: generation uses 50 reverse steps in all seven cases. Retraining may change a checkpoint; the released checkpoint asset is the reference for the reported sampling runs.

## Generate and evaluate

```bash
python -m dygraft run --dataset adult --seed 20261601 \
  --output results/adult_20261601 --evaluate
```

The output table is `synthetic.csv`. Intermediate folders contain the candidate pool, anchor, support trajectory, train-only calibration, and refinement. `run.json` records stage timings and `commands.jsonl` records the executed commands. Evaluation writes quality/utility metrics and the separate SQL workload and fidelity scores.

Use `--checkpoint PATH` and optionally `--model-config PATH` for a retrained model. The frozen-loop trainer writes `ema_model_epoch_8000.pt`; the upstream trainer writes its selected `best_ema_model_*.pt` under `tabdiff/ckpt/<dataset>/dygraft/`.

Default generates 16,384 candidates; the other datasets generate 8,192. Selection produces 4,096 rows. Configuration files contain the five main-run seeds and three basic-ablation seeds for each dataset.

## Controlled comparisons

Direct backbone and backbone with the same selector:

```bash
python -m dygraft run --dataset adult --seed 20261601 \
  --arm backbone --output results/adult_backbone --evaluate
python -m dygraft run --dataset adult --seed 20261601 \
  --arm backbone-selector --output results/adult_backbone_selector --evaluate
```

For a CPU-only screening comparison, use the **same candidate CSV and seed** for both commands:

```bash
python -m dygraft select --dataset adult --seed 20261601 \
  --pool results/adult_20261601/pool/samples/dense_factor.csv \
  --screening off --output results/adult_unscreened
python -m dygraft select --dataset adult --seed 20261601 \
  --pool results/adult_20261601/pool/samples/dense_factor.csv \
  --screening on --output results/adult_screened
```

Screening uses batched approximate scores to shortlist proposals, followed by exact guard evaluation for retained candidates. It is not a universal exactness guarantee. Compare selected indices and output tables along with latency. `--stage anchor` and `--stage support` expose the two intermediate selection ablations. Low-level sampling options, including routing and persistence controls, remain available through `python -m dygraft.sampling --help`.

To evaluate multiple tables under one protocol:

```bash
python -m dygraft evaluate --dataset adult --seed 20261601 \
  --sample Screened=results/adult_screened/synthetic.csv \
  --sample Unscreened=results/adult_unscreened/synthetic.csv \
  --output results/adult_comparison
```

Shape/Trend use SDMetrics; alpha/beta and task utility use the experiment evaluators. Utility is ROC-AUC for classification and R² for Insurance. SQL workload construction uses the training split, excludes graph-covered attribute pairs, and compares answers against the held-out split. Selection and policy calibration do not optimize held-out metrics. Cross-method latency measurements should use identical hardware, candidate/output sizes, reverse steps, seeds, and CPU thread limits; report direct-backbone and oversampled-selector costs separately.

## Implementation map

| Component | Modules |
|---|---|
| Data preparation and encoding | `data.py`, `preprocessing.py`, `encoding.py` |
| Frozen backbone | `tabdiff/models/`, `tabdiff/modules/` |
| Verified factors and routing | `graph.py`, `predicates.py`, `sampling.py`, `preview.py` |
| Typed checks, evidence calibration, role assignment | `graph_construction.py` |
| Numerical/categorical messages | `distribution_transport.py`, `categorical_guidance.py` |
| Distribution anchor | `anchoring.py` |
| Support restoration and screening | `support.py`, `coverage_index.py`, `screened_support.py`, `quality_cache.py` |
| Train-only calibration | `calibration.py`, `train_coverage.py` |
| Graph refinement | `refinement.py`, `factor_statistics.py`, `quality_constraints.py`, `cached_refinement.py` |
| Evaluation | `evaluation.py`, `evaluation_mixed.py`, `query_fidelity.py` |

Modules in this table are under `dygraft/` unless otherwise indicated. Frozen graph inputs and edge-role permissions are provided in `configs/graphs/`; generation does not call an external language-model API. `graph_construction.py` exposes the deterministic `typecheck`, `verify`, and `select_after_calibration` functions for inspecting or rebuilding a graph from supplied candidate dictionaries. Historical experiment launchers, research notes, logs, credentials, verification outputs, and unused exploratory branches are excluded from the source release. Third-party comparison methods should be run from their respective official implementations rather than interpreted as implementations supplied here.

## Acknowledgments and license

The backbone and associated training utilities derive from [TabDiff](https://github.com/MinkaiXu/TabDiff). Its MIT copyright and license are retained in `LICENSE`. Please cite TabDiff when using this implementation, in addition to DyGRAFT when its publication information is available.
