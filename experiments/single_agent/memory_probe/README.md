# Memory probe

Run from the repository root:

```sh
flag-game probe memory --backend scripted --model gpt-4o --truth-country Peru --m 1 --replicates 1 --memory-counts 0,8 --no-plots --render-scale 1 --out runs/memory_smoke
flag-game probe memory --backend openai --model gpt-5.6-terra --reasoning-effort none --temperature 0.2 --replicates 3 --out runs/terra_memory_conflict
flag-game probe memory --help
```
