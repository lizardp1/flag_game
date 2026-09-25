# Social experiments

Run from the repository root. Edit the YAML files to set N, model counts, alpha,
bandwidth, memory, rounds, and seeds. Model counts must sum to N.
Presets use the scripted backend; see `docs/reproduce.md` for API runs.

```sh
flag-game run --config experiments/social/pairwise.yaml
flag-game run --config experiments/social/broadcast.yaml
flag-game run --config experiments/social/manager.yaml
flag-game sweep --config experiments/social/alpha_composition.yaml
flag-game analyze --runs runs/alpha_composition --out runs/alpha_composition/analysis
flag-game plot --summary runs/alpha_composition/analysis/summary.csv --out runs/alpha_composition/analysis/accuracy.png --font-size 12
```
