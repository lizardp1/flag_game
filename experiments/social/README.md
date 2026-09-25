# Social experiments

`pairwise.yaml`, `broadcast.yaml`, and `manager.yaml` use the same schema. Change the configuration, not the Python implementation, for N, composition, alpha, bandwidth, memory, rounds, seeds, and crop geometry. Set integer model counts summing to N; manager adds a separately configured blind agent.

- `alpha_composition.yaml`: the 5 x 9 broadcast alpha/composition grid.
- `protocols.yaml`: one matched configuration across the three protocols (validate saved crops before matched statistical analysis).
- `bandwidth.yaml`: all three protocols and all three bandwidths.

All default to scripted. For paid sweeps edit the base preset's backend and exact seed list after preflight. Keep a broadcast-only refresh separate from full protocol sweeps. Sweep cells have deterministic hashes and retain resolved settings; changing only figure appearance does not create cells.

- `population.yaml`: explicit N/composition cases for both homogeneous models at N=4,8,16,32,64,128.

After execution, separate analysis and styling:

```sh
flag-game analyze --runs runs/alpha_composition --out runs/alpha_composition/analysis
flag-game plot --summary runs/alpha_composition/analysis/summary.csv --out runs/alpha_composition/analysis/accuracy.png --font-size 12
```

Repeated plotting replaces the same chart. Analysis rejects duplicate configuration/seed observations rather than silently double-counting reruns.
