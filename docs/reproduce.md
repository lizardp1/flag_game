# Reproduction

## Offline path

Install with `python -m pip install -e '.[dev,paper]'`. Run `python -m unittest discover -s tests -v`, then the README's figure and theory commands. Social presets use scripted inference, so no credentials are required.

Use `flag-game prompts --config experiments/social/broadcast.yaml` to inspect executable prompt templates and `flag-game run ... --dry-run` to inspect resolved settings before execution.

## Paid runs

Store the OpenAI key only in local ignored `.env.local`, with mode 0600. Do not commit or print keys. Load it only in the shell command that needs it:

```sh
set -a; source .env.local; set +a
```

Before a paid rerun, make a no-inference authentication request and print only its HTTP status:

```sh
python - <<'PY'
import os, urllib.request, urllib.error
request = urllib.request.Request('https://api.openai.com/v1/models', headers={'Authorization': 'Bearer '+os.environ['OPENAI_API_KEY']})
try:
    with urllib.request.urlopen(request) as response: print(response.status)
except urllib.error.HTTPError as error:
    print(error.code)
PY
```

Before a large paid sweep, run a small real-model smoke trial into a fresh output root and inspect prompt examples, per-call `debug/**/calls.jsonl`, protocol logs, memory snapshots, API usage, and trial manifest. No paid calls were needed for repository cleanup.

```sh
flag-game run --config experiments/social/broadcast.yaml --set backend=openai --set N=2 --set 'composition={gpt-4o: 2}' --set rounds=2 --set output_root=runs/openai_smoke
```

Use the full alpha/composition sweep and the broadcast-only protocol slice separately. Do not refresh unchanged pairwise or manager data merely to update broadcast figures. Put every scientific rerun in a fresh root; plotting an existing run reuses that run's data.

## Run products

Each social trial stores `experiment.json` (status, common/resolved settings, seed, source hash), `prompt_examples.json`, the engine's `trial_manifest.json`, protocol logs, summary, and API usage. OpenAI per-call audits retain messages/responses and provider-returned model IDs. Internal logs may identify models; peer-facing prompts must not.

`--resume` skips only identical completed trials. Partial/failed data remains intact for audit; use a fresh output directory to retry. Logs are kept locally in ignored `runs/`; publish selected data through a separately reviewed release/data archive.

## Historical data versus new experiments


## Tested dependency environment

`requirements-lock.txt` records the successful clean Python 3.10/macOS installation. For that tested dependency set, install `python -m pip install -r requirements-lock.txt` followed by `python -m pip install --no-deps -e .`. Other platforms may resolve different compatible wheels. `pyproject.toml` remains the supported dependency specification.
