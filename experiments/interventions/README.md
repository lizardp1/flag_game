# Crop interventions

Replace an observer’s crop and measure how decisions change under the pairwise protocol.

```sh
nnd-flag-game run --config experiments/interventions/crop_patch.yaml --out runs/crop_patch_smoke --backend scripted --seed 0
```

Set `engineered_crop_agent_id` to choose the observer and `engineered_crop_preference` to select `best` or `worst` according to the catalog’s crop-compatibility score.

## Temporal reach

Compute earliest-arrival times through a saved communication schedule:

```sh
python experiments/interventions/temporal_reach.py --interactions <run>/interactions.jsonl --N 8 --out <run>/temporal_reach.json
```

Add `--valid-only` to include only valid messages. The calculation measures potential information reach through the schedule.

Crop-patching methods and population chart data are in `paper/crop_patching/`.
