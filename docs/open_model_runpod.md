# Open-Model RunPod Smoke Test

Use this after `docs/runpod_pod.md` is working. The goal is to validate one
local vision-language model on the GPU Pod before wiring it into the full paper
runner.

## Recommended First Model

Start with:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

It is small enough for a 48 GB GPU with headroom, supports image+text prompts in
Transformers, and is a useful first target for later internal-representation
capture.

## Install Open-Model Dependencies

Use a RunPod PyTorch/CUDA template so `torch` is already installed for the
machine's CUDA version.

```bash
cd /workspace/flag_game
git checkout codex-runpod-pod-bootstrap
git pull --ff-only origin codex-runpod-pod-bootstrap

source .venv/bin/activate
python -m pip install -r requirements-open-models.txt
```

If you did not use a PyTorch template, install the correct CUDA build of PyTorch
first, then rerun the command above.

Qwen's processor requires `torchvision`. If you installed the open-model
requirements before this line was added, rerun:

```bash
source .venv/bin/activate
python -m pip install -r requirements-open-models.txt
```

If your PyTorch template has a custom Torch build and pip tries to replace it,
install only the missing package instead:

```bash
python -m pip install torchvision --no-deps
```

## Cache Location

Keep Hugging Face downloads on the persistent `/workspace` volume:

```bash
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
mkdir -p "$HF_HOME"
```

If a model is gated, also set:

```bash
export HF_TOKEN=...
```

## Dry Run

This checks the repo-side prompt, crop rendering, and output paths without
loading the model:

```bash
python scripts/run_qwen_vl_smoke.py --dry-run --out runs/qwen_vl_smoke_dry
```

Or use the RunPod wrapper:

```bash
./scripts/runpod_qwen_vl_smoke.sh --dry-run --out runs/qwen_vl_smoke_dry
```

## Real Qwen Smoke

The default smoke uses `m=3`, so the model must return both a country and a
one-sentence reason:

```json
{"country":"<one allowed country>","reason":"<one sentence>"}
```

```bash
python scripts/run_qwen_vl_smoke.py \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --out runs/qwen_vl_smoke
```

Equivalent wrapper command:

```bash
./scripts/runpod_qwen_vl_smoke.sh \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --out runs/qwen_vl_smoke
```

To reproduce the earlier country-only smoke, add `--m 1`.

The script writes:

- `runs/qwen_vl_smoke/artifacts/truth_flag.png`
- `runs/qwen_vl_smoke/artifacts/agent_crop.png`
- `runs/qwen_vl_smoke/prompt.txt`
- `runs/qwen_vl_smoke/stimulus.json`
- `runs/qwen_vl_smoke/result.json`

## Optional Hidden-State Shape Check

This does one extra forward pass with `output_hidden_states=True` and saves only
shape metadata in `result.json`. It is the first tiny bridge toward mech interp.

```bash
python scripts/run_qwen_vl_smoke.py \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --out runs/qwen_vl_smoke_hidden \
  --activation-summary
```

If you hit memory pressure, rerun without `--activation-summary`.

## Pairwise Flag-Game Smoke

After the one-image smoke passes, run the same Qwen model through the actual
pairwise `nnd.flag_game` runner. This uses `m=3`, so agent messages include a
country plus a one-sentence reason.

```bash
python -m nnd.cli run \
  --config configs/open_models/qwen2_5_vl_7b_pairwise_smoke.yaml \
  --out runs/qwen_pairwise_smoke \
  --backend transformers_vlm
```

The config keeps the run intentionally tiny:

- `N=2`, `T=2`, `H=2`
- `tile_height=6`, which avoids the default-seed case where both agents see
  visually identical stripes
- `probe_workers=1` to serialize local GPU generation
- `activation_capture.enabled=true`, scoped to the initial `t=0`, `m=3` probe
- `output.make_plots=false` to avoid spending time on plotting during smoke tests

The backend is selected by:

```yaml
backend: transformers_vlm
model: Qwen/Qwen2.5-VL-7B-Instruct
```

Useful environment switches for the local Transformers backend:

```bash
export NND_TRANSFORMERS_DTYPE=bfloat16
export NND_TRANSFORMERS_DEVICE_MAP=auto
export NND_TRANSFORMERS_ATTN_IMPLEMENTATION=auto
```

If you install `flash-attn`, you can try:

```bash
export NND_TRANSFORMERS_ATTN_IMPLEMENTATION=flash_attention_2
```

Expected outputs include the normal paper-run artifacts under
`runs/qwen_pairwise_smoke`, local crop PNGs under the backend debug directory,
and compact activation shards for the initial probe:

```text
runs/qwen_pairwise_smoke/
runs/qwen_pairwise_smoke/activations/index.jsonl
runs/qwen_pairwise_smoke/activations/tensors/
runs/qwen_pairwise_smoke/debug/Qwen_Qwen2.5-VL-7B-Instruct/prepared_crops/
```

## First Activation Dataset

For linear probes, collect a small batch of initial probes only. This avoids
spending GPU time on full social interaction rounds while you are just checking
that representation capture works.

```bash
python -m nnd.cli batch \
  --config configs/open_models/qwen2_5_vl_7b_pairwise_smoke.yaml \
  --out runs/qwen_activation_t0_batch \
  --backend transformers_vlm \
  --start-seed 0 \
  --num-seeds 8 \
  --probe-workers 1 \
  --seed-workers 1 \
  --override T=0
```

Then train a first per-layer linear probe:

```bash
python scripts/analyze_activation_linear_probe.py \
  --runs runs/qwen_activation_t0_batch \
  --target truth_country \
  --feature last_prompt_token \
  --out runs/qwen_activation_t0_batch/linear_probe_truth_country
```

Useful alternate targets:

```bash
python scripts/analyze_activation_linear_probe.py \
  --runs runs/qwen_activation_t0_batch \
  --target informativeness_label \
  --feature mean_prompt \
  --out runs/qwen_activation_t0_batch/linear_probe_informativeness
```

The probe script writes:

- `linear_probe_results.csv`
- `summary.json`
- `samples.csv`
