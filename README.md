# Flag Game

Agents identify a hidden flag from private crops and exchange messages under
pairwise, broadcast, or manager protocols. The repository includes configurable
experiments, readable prompts, isolated memory and vision probes, theory, and
paper figures.

## Install

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

## Run

```sh
flag-game run --config experiments/social/pairwise.yaml
flag-game run --config experiments/social/broadcast.yaml --dry-run
flag-game prompts --config experiments/social/manager.yaml
flag-game sweep --config experiments/social/alpha_composition.yaml --dry-run
```

Presets use a scripted backend with no API charges. Set the protocol, population
size, model composition, social-evidence alpha, message bandwidth, memory, and
seeds in the configs. See [controls](docs/protocols.md) and
[run instructions](docs/reproduce.md) for real-model runs and preflight.
Scripted runs check software behavior, not scientific model performance.

## Files

- `experiments/`: social experiments, single-agent probes, and intervention examples.
- `nnd/`: shared code and protocol engines.
- `prompts/`: readable prompts generated from executable code.
- `theory/`: accepted model, equations, inputs, and builders.
- `paper/figures/final/`: complete main-paper figures, numbered fig1–fig8.
- `paper/figures/generated/`: reproducible chart components.

```sh
bash paper/figures/build.sh
python theory/validate.py
python -m unittest discover -s tests -v
```

The research code uses the MIT license. Retain third-party notices in
`LICENSING.md`.
