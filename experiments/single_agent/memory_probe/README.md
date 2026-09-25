# Memory-conflict probe

One fixed private crop plus synthetic FIFO memory; no communicating population. Controls: models, m in {1,3}, H, false-memory counts, truth country, crop ambiguity, lure compatibility, replicates, seed, image settings. The executable implementation is `nnd/probes/memory.py`.

```sh
flag-game probe memory --backend scripted --model gpt-4o --truth-country Peru --m 1 --replicates 1 --memory-counts 0,8 --no-plots --render-scale 1 --out runs/memory_smoke
```

Use `flag-game probe memory --help` for the full interface, `--dry-run` to generate stimuli and the trial plan, and `--report-only` to re-analyze existing data. Canonical paper charts use `paper/figures/data/raw/memory_conflict_probe_*`; new runs do not replace those inputs automatically. Existing completed rows are skipped by this legacy probe runner; `--force` intentionally reruns them. Prefer fresh run roots.
