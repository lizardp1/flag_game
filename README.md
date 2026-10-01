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

For an OpenAI run, load your key from the local, git-ignored `.env.local`,
then choose the model, population, seeds, and controls:

```sh
(
  set -a
  source .env.local
  set +a

  MODEL=gpt-4o
  N=16
  SEEDS='[0,1,2]'

  flag-game run --config experiments/social/pairwise.yaml \
    --set backend=openai \
    --set "N=$N" \
    --set "composition={${MODEL}: ${N}}" \
    --set "seeds=$SEEDS" \
    --set country_pool=stripe_expanded_24 \
    --set rounds=32 \
    --set "probe_every=$N" \
    --set early_stop_window=5 \
    --set message_bandwidth=3 \
    --set memory_capacity=8 \
    --set temperature=0.2 \
    --set workers=8 \
    --set seed_workers=2 \
    --set "output_root=runs/${MODEL}_N${N}_$(date +%Y%m%d_%H%M%S)"
)
```

This runs up to 32 rounds per seed, probes every N interactions, and stops
when the same country has full consensus for five consecutive probes.
For a mixed population, set `composition` to model counts that sum to N.
OpenAI pairwise runs also accept `--set reasoning_effort=none` for models
that support it; choose sampling parameters supported by your model.
Add `--dry-run` to inspect the configuration without API calls.

## Files

- `experiments/`: social experiments, single-agent probes, and intervention examples.
- `nnd/`: shared code and protocol engines.
- `prompts/`: readable prompts generated from executable code.
- `theory/`: model implementation, inputs, and figure builders.
- `paper/figures/final/`: complete main-paper figures, numbered fig1–fig8.
- `paper/figures/generated/`: reproducible chart components.

```sh
bash paper/figures/build.sh
python theory/validate.py
python -m unittest discover -s tests -v
```

The research code uses the MIT license. Retain third-party notices in
`LICENSING.md`.
