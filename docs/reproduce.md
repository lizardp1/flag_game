# Reproduce

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Baseline tests

```bash
python -m unittest discover -s tests
```

## Final Paper Charts

Exact final-paper run metadata is recorded in `configs/final_paper_reproduction.yaml`. It includes model IDs, API settings, trial counts, seed inventories, data sources, and figure reproduction commands. The prompt template snapshot is in `configs/final_paper_prompt_templates.md`.

The final chart bundle is self-contained under `paper/final_charts/`:

```bash
cd paper/final_charts
./scripts/rebuild_final_charts.sh
```

The bundle contains final figure files in `paper/exports/figures/`, source tables in `paper/exports/data/`, slimmed run data in `results/`, and the plotting scripts in `paper/`. Full per-seed image artifact dumps are not part of the bundle; only the stimulus PNGs and compact plotted summaries needed by final panels are retained.

## Pairwise control run

```bash
python -m nnd.cli run --config configs/pairwise_control.yaml --out runs/pairwise_control --trials 1
```

## Group-wise development status

Group-wise helper modules exist under `nnd/groupwise/`, but full CLI integration is still pending.
