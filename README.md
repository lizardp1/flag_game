# memetic-drift-groupwise

Standalone research repo scaffold for extending Neutral Naming Drift from pairwise interactions to group-wise interactions.

## Scope

- Baseline pairwise engine copied from `nnd/` as the reproducible starting point.
- New `nnd.groupwise` module for group sampling, memory updates, and group coordination metrics.
- Baseline and extension tests under `tests/`.
- Reproducibility docs and initial configs under `docs/` and `configs/`.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests
```

## Final Paper Reproducibility

The exact final-chart reproduction metadata lives in:

- `configs/final_paper_reproduction.yaml`: model IDs, API settings, trial counts, seed inventories, data sources, and figure commands.
- `configs/final_paper_prompt_templates.md`: exact prompt templates used by the final Flag Game runs.
- `paper/final_charts/`: final paper figures, plotting scripts, source data, and `scripts/rebuild_final_charts.sh`.

To rebuild the final paper figures from the bundled data:

```bash
cd paper/final_charts
./scripts/rebuild_final_charts.sh
```

## Theory Docs

- `docs/theory/qsg_foundation.tex`
- `docs/theory/qsg_foundation.pdf`
- `docs/theory/README.md`

## Current status

- Pairwise pipeline is runnable via `python -m nnd.cli run ...`.
- Group-wise support is scaffolded at module level (`nnd/groupwise/`) and not fully wired into `nnd.cli` yet.

## Suggested branch policy

- `main`: stable branch for reproducible runs.
- `feature/groupwise-*`: active extension work.
- Tag milestones used in figures and paper drafts.
