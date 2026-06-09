# RunPod Pod Workflow

This repo is designed to run on a regular RunPod Pod: clone the GitHub repo into
`/workspace`, pull new changes, bootstrap the Python environment, and run the
paper/research commands locally on the GPU machine.

## First-Time Pod Setup

Start from a RunPod Pod with a Python/CUDA or PyTorch template. A network volume
mounted at `/workspace` is useful so the repo, virtualenv, model cache, and
result directories survive pod restarts.

```bash
cd /workspace
git clone https://github.com/lizardp1/flag_game.git
cd flag_game
./scripts/runpod_bootstrap.sh
./scripts/runpod_smoke_test.sh
```

If you want to test an unmerged branch from this Codex workspace:

```bash
cd /workspace
git clone --branch codex-runpod-pod-bootstrap https://github.com/lizardp1/flag_game.git flag_game
cd flag_game
./scripts/runpod_bootstrap.sh
./scripts/runpod_smoke_test.sh
```

After the branch is merged, use `main`:

```bash
git checkout main
git pull --ff-only origin main
```

## Pull Latest Changes

From an existing clone:

```bash
cd /workspace/flag_game
./scripts/runpod_pull_and_smoke.sh
```

To follow a specific branch:

```bash
cd /workspace/flag_game
BRANCH=codex-runpod-pod-bootstrap ./scripts/runpod_pull_and_smoke.sh
```

The script fetches/pulls the requested branch, refreshes `.venv`, and runs the
no-network scripted smoke test.

## Rebuild Final Paper Charts

The bundled final charts are rebuilt from the compact paper data under
`paper/final_charts`.

```bash
cd /workspace/flag_game
source .venv/bin/activate
cd paper/final_charts
./scripts/rebuild_final_charts.sh
```

## Run a Small Pairwise Smoke Experiment

This uses the deterministic scripted backend, so it checks the runner without
calling paid model APIs:

```bash
cd /workspace/flag_game
source .venv/bin/activate
python -m nnd.cli run \
  --config configs/flag_game/stripe_easy_v1.yaml \
  --out runs/pairwise_scripted_smoke \
  --backend scripted \
  --override output.make_plots=false
```

## Notes

- Keep large outputs under `results/`, `runs/`, or another ignored directory.
- Keep API keys and Hugging Face tokens in environment variables, not files
  committed to git.
- Later open-model/mech-interp work should use this Pod-local path so we can run
  Transformers/Accelerate/NNsight and capture internal activations directly.
