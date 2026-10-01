# Crop intervention

Run from the repository root:

```sh
nnd-flag-game run --config experiments/interventions/crop_patch.yaml --out runs/crop_patch_smoke --backend scripted --seed 0
```

Set `engineered_crop_agent_id` and `engineered_crop_preference` (`best` or `worst`,
by catalog crop compatibility) in the YAML.

Calculate temporal reach from a saved run:

```sh
python experiments/interventions/temporal_reach.py --interactions runs/crop_patch_smoke/interactions.jsonl --N 8 --out runs/crop_patch_smoke/temporal_reach.json
```
