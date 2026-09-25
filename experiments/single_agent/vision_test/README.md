# Visual-only paired-crop test

Single agents identify identical crops without social memory. Controls: model list, country pool, truth countries, crops per informativeness label, seed, m in {1,3}, image settings. Implementation: `nnd/probes/vision.py`.

```sh
flag-game probe vision --backend scripted --model gpt-4o --truth-country Peru --limit-pairs 2 --render-scale 1 --out runs/vision_smoke
```

Use `flag-game probe vision --help`, `--dry-run` to prepare a plan, or `--report-only` to rebuild tables. Qualitative adjudication is a separate saved input, not silently recomputed model ground truth. Paper inputs live in `paper/figures/data/raw/visual_only_paired_crop_audit/`. New probe plots stay with their run until explicitly promoted.
