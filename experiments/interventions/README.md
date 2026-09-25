# Crop interventions and social circuit attribution

Accepted manuscript text, figure assets, and the compact population chart data live in `paper/crop_patching/`. They are separate from the binary theory and isolated-agent probes.

The existing pairwise engine supports replacing one observer's crop with the best/worst crop under the catalog's compatibility diagnostic:

```sh
nnd-flag-game run --config experiments/interventions/crop_patch.yaml --out runs/crop_patch_smoke --backend scripted --seed 0
```

This preset tests the intervention mechanism; it is **not** the paper's exact Germany A4 intervention. The paper's informative crop was selected by real-model probe accuracy, not the catalog diagnostic. Its full paired logs and temporal attribution pipeline remain in the source research repository under `rebuttal/causal_intervention/`. The compact accepted data is preserved, but this cleanup does not claim exact paid causal replay from the public package. A production replication must import the selected crop coordinates, original schedule, and cohort manifest, rather than select a new 'best' crop.

Use `python experiments/interventions/temporal_reach.py --interactions <run>/interactions.jsonl --N 8 --out <run>/temporal_reach.json` to compute first-arrival times from any pairwise schedule without API calls. It describes opportunities for transmission, not evidence of actual belief transmission. Use only valid contacts with `--valid-only` to exclude parse-invalid messages; default includes all scheduled contacts.
