# Reproducibility guide

## Environment

The original experiments used the server environment recorded in `environment/pip_freeze.txt`. Recreate an isolated environment rather than assuming the package is directly portable across CUDA/PyTorch versions.

## Required external artifacts

- dataset directories;
- trained TabDiff checkpoints and config pickles;
- graph JSON files;
- optional published-baseline outputs.

Original paths are preserved in Slurm scripts, provenance files, and `manifests/OMITTED_ARTIFACTS.csv`.

## Reproduction order

1. Restore dataset and checkpoint paths or update a copied launcher.
2. Run the relevant smoke job.
3. Generate paired `B/G` pools through Slurm.
4. Build frozen C3 using the same pool and seed.
5. Run graph-proximal repair and graph audit.
6. Run the common evaluator.
7. Run the dependent summary job.

## Exact final pipeline

- `slurm/902_graph_deficit_gate_smoke.sbatch`
- `slurm/903_confirm_graph_deficit_gate.sbatch`
- `slurm/904_summarize_graph_deficit_gate_confirmation.sbatch`

## Validation checks

- array tasks complete with exit code `0:0`;
- unconditional hashes match within paired comparisons where required;
- selected indices are unique and within pool bounds;
- selection metadata reports feasibility or an explicit deficit-gate no-op;
- graph audit is external to the selection ranking;
- summary count matches the expected seeds;
- screen and confirmation seeds are disjoint.

