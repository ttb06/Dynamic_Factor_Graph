# Results quick reference

| Dataset | Method/evidence | Shape | Trend | Gain vs dynamic `G` | Gain vs frozen C3 | Local records | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Adult | deficit-gated graph-proximal | 0.860949 | 2.017464 | +0.371348/+0.521812 | +0.008133/+0.024268 | 4/5 | comprehensive |
| Shoppers | quality-first proximal | 1.102046 | 1.803424 | +0.389162/+0.234395 | -0.024145/-0.030366 | 3/5 | coherent |
| Magic | quality-first proximal | 0.956562 | 2.334173 | +0.246281/-0.033541 | +0.008858/+0.007241 | 2/5 | reject strict gate |
| Default | deficit-gated held-out | 0.964895 | 1.435227 | +0.376324/+0.027127 | -0.007101/-0.000045 | 2/5 | reject |
| Magic | frozen `proj+gt4` | 1.009621 | 1.060297 | +0.318999/+0.887710 | n/a | 3/5 | comprehensive C3 |
| Diabetes | frozen `hybrid+int20` | 1.353019 | 8.165711 | +0.319212/+1.137281 | n/a | 3/5 | primary confirmed |
| Shoppers | frozen `proj+gt4` | 1.097399 | 2.096839 | +0.543518/+0.107746 | n/a | 2/5 | primary confirmed |

The graph repair is the final proposed extension; frozen C3 remains a strong independent ablation and should not be hidden.

