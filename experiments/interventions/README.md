# Crop interventions and social circuit attribution

Accepted manuscript text, figure assets, and the compact population chart data live in `paper/crop_patching/`. They are separate from the binary theory and isolated-agent probes.

The existing pairwise engine supports replacing one observer's crop with the best/worst crop under the catalog's compatibility diagnostic:

```sh
nnd-flag-game run --config experiments/interventions/crop_patch.yaml --out runs/crop_patch_smoke --backend scripted --seed 0
```

The preset selects a crop using the catalog compatibility diagnostic. The paper's Germany A4 intervention instead used a crop selected through model probes, so this preset demonstrates the intervention mechanism rather than replaying that experiment.

Use `python experiments/interventions/temporal_reach.py --interactions <run>/interactions.jsonl --N 8 --out <run>/temporal_reach.json` to compute first-arrival times from any pairwise schedule without API calls. It describes opportunities for transmission, not evidence of actual belief transmission. Use only valid contacts with `--valid-only` to exclude parse-invalid messages; default includes all scheduled contacts.
