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

## Visual-Only Perception Tests

If the model keeps guessing one country, test raw vision before running more
flag-game batches. This removes the country list and asks only about colors,
stripe orientation, stripe count, and stripe order.

Small color-only pass:

```bash
python scripts/run_qwen_visual_perception_tests.py \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --suite colors \
  --pixel-sizes 24x16,48x32,75x150,150x100,300x200 \
  --out runs/qwen_visual_colors_7b
```

Full color-plus-stripe size sweep:

```bash
python scripts/run_qwen_visual_perception_tests.py \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --suite all \
  --out runs/qwen_visual_perception_7b
```

Compare model scales with the same visual-only battery:

```bash
python scripts/run_qwen_visual_perception_tests.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --suite all \
  --out runs/qwen_visual_perception_3b_7b
```

For a fast local bookkeeping check that does not load a model:

```bash
python scripts/run_qwen_visual_perception_tests.py \
  --backend oracle \
  --max-tests 8 \
  --out runs/qwen_visual_perception_oracle
```

The script writes:

- `results.csv`: one row per model call, including raw JSON and correctness
- `size_summary.csv`: accuracy by model, task type, and image size
- `color_summary.csv`: per-color accuracy and predicted colors
- `stripe_summary.csv`: per-pattern orientation/count/order accuracy
- `breakpoints.csv`: largest-to-smallest first size below the accuracy threshold
- `artifacts/`: the exact synthetic images sent to the model

Interpretation:

- If solid colors fail, the model/interface is not reliable enough for crop
  geometry yet.
- If colors pass but stripes fail, the bottleneck is likely spatial binding:
  stripe count, orientation, or left-to-right/top-to-bottom ordering.
- If large images pass and small images fail, tune crop size/render scale before
  interpreting social convergence.
- If these tests pass but country guesses still collapse to France, treat the
  issue as country-prior, prompt, answer-schema, or candidate-list bias.

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
- `activation_capture.enabled=true`, scoped to all probe calls for geometry analysis
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

For representation geometry across communication rounds:

```bash
python -m nnd.cli run \
  --config configs/open_models/qwen2_5_vl_7b_pairwise_smoke.yaml \
  --out runs/qwen_geometry_smoke \
  --backend transformers_vlm
```

Then compute cross-agent cosine similarity and per-agent temporal drift:

```bash
python scripts/analyze_activation_geometry.py \
  --runs runs/qwen_geometry_smoke \
  --feature last_prompt_token \
  --out runs/qwen_geometry_smoke/activation_geometry
```

For a small multi-seed geometry dataset:

```bash
python -m nnd.cli batch \
  --config configs/open_models/qwen2_5_vl_7b_pairwise_smoke.yaml \
  --out runs/qwen_geometry_batch \
  --backend transformers_vlm \
  --start-seed 0 \
  --num-seeds 8 \
  --probe-workers 1 \
  --seed-workers 1
```

```bash
python scripts/analyze_activation_geometry.py \
  --runs runs/qwen_geometry_batch \
  --feature last_prompt_token \
  --out runs/qwen_geometry_batch/activation_geometry
```

For optional linear probes, collect a small batch of initial probes only. This
avoids spending GPU time on full social interaction rounds while you are just
checking that representation labels are recoverable.

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

## Model-Scale Capability Sweep

If a model collapses toward one country such as France, first test scale on the
initial visual probes before communication can spread a wrong answer.

Default sweep, practical for a 48 GB GPU:

```bash
./scripts/run_qwen_scale_capability_sweep.sh
```

This runs:

```text
Qwen/Qwen2.5-VL-3B-Instruct
Qwen/Qwen2.5-VL-7B-Instruct
```

The default sweep uses `T=0`, `N=8`, and `num_seeds=8`, giving 64 initial crop
predictions per model.

For a quick smoke:

```bash
NUM_SEEDS=2 N_AGENTS=4 ./scripts/run_qwen_scale_capability_sweep.sh
```

For an 80 GB GPU, optionally try 32B:

```bash
MODELS="Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen2.5-VL-32B-Instruct" \
OUT_ROOT=runs/qwen_scale_capability_3b_7b_32b \
./scripts/run_qwen_scale_capability_sweep.sh
```

The summarizer writes:

```text
runs/qwen_scale_capability/scale_summary/model_scale_metrics.csv
runs/qwen_scale_capability/scale_summary/initial_probe_country_distribution.csv
runs/qwen_scale_capability/scale_summary/initial_probe_agent_rows.csv
```

Key columns:

- `initial_agent_accuracy`
- `initial_agent_france_rate`
- `initial_agent_predicted_compatible_rate`
- `france_when_france_incompatible_rate`

If larger models reduce `france_when_france_incompatible_rate`, that points to a
capability/scale issue. If every scale still says France when France is
incompatible, treat it as a prompt/interface bias or crop-resolution issue
before interpreting the geometry.
