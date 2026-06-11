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

Qwen's processor requires `torchvision`. Kimi-VL's remote-code tokenizer path
requires `tiktoken` and `blobfile`. If you installed the open-model
requirements before those lines were added, rerun:

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
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export HF_ASSETS_CACHE=/workspace/.cache/huggingface/assets
export HF_XET_CACHE=/workspace/.cache/huggingface/xet
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub
export TMPDIR=/workspace/.cache/tmp
export HF_XET_NUM_CONCURRENT_RANGE_GETS=2
export HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1
mkdir -p "$HF_HOME" "$HF_XET_CACHE" "$TMPDIR"
```

If a model is gated, also set:

```bash
export HF_TOKEN=...
```

The Qwen wrapper scripts source `scripts/runpod_cache_env.sh`, which defaults
these caches to `/workspace/.cache` when `/workspace` is writable. If a failed
download mentions `/tmp/nnd_matplotlib_cache/huggingface`, clean the partial
temporary cache and rerun from a shell with workspace cache variables:

```bash
df -h /workspace /tmp
rm -rf /tmp/nnd_matplotlib_cache/huggingface

source scripts/runpod_cache_env.sh
echo "$HF_HOME"
echo "$HF_HUB_CACHE"
echo "$HF_XET_CACHE"
echo "$TMPDIR"
```

If `/workspace` is also tight, inspect cached models:

```bash
du -h -d 2 /workspace/.cache/huggingface 2>/dev/null | sort -h | tail -40
```

If a download fails inside `xet_get` with an error like `Internal Writer Error`
or `Background writer channel closed`, pre-download the model once with
conservative Hugging Face/Xet settings, then rerun the visual sweep from cache:

```bash
cd /workspace/flag_game
source .venv/bin/activate
source scripts/runpod_cache_env.sh

./scripts/runpod_hf_download_model.sh llava-hf/llava-v1.6-mistral-7b-hf
```

For Kimi-VL:

```bash
./scripts/runpod_hf_download_model.sh moonshotai/Kimi-VL-A3B-Instruct
```

The downloader uses `snapshot_download(..., max_workers=1)` by default. To try
two workers:

```bash
HF_SNAPSHOT_MAX_WORKERS=2 ./scripts/runpod_hf_download_model.sh llava-hf/llava-v1.6-mistral-7b-hf
```

Before loading a new open VLM, run:

```bash
./scripts/runpod_model_preflight.sh
```

If the RunPod web terminal disconnects or reopens as a blank terminal during a
model run, treat it as a likely process/container kill. Capture logs to a file:

```bash
MODELS="Qwen/Qwen3-VL-4B-Instruct" \
OUT=runs/qwen3_visual_4b \
./scripts/run_qwen_visual_model_sweep.sh 2>&1 | tee runs/qwen3_visual_4b.log
```

If even a foreground command clears the terminal, launch it in the background
and tail the log from a fresh terminal:

```bash
nohup bash -lc 'cd /workspace/flag_game && source .venv/bin/activate && MODELS="llava-hf/llava-v1.6-mistral-7b-hf" OUT=runs/llava_visual_colors ./scripts/run_qwen_visual_model_sweep.sh' \
  > runs/llava_visual_colors.log 2>&1 &

tail -f runs/llava_visual_colors.log
```

For an even smaller Qwen3 check:

```bash
MODELS="Qwen/Qwen3-VL-2B-Instruct" \
OUT=runs/qwen3_visual_2b \
./scripts/run_qwen_visual_model_sweep.sh 2>&1 | tee runs/qwen3_visual_2b.log
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

Focused color-confusion sweep:

```bash
./scripts/run_qwen_visual_model_sweep.sh
```

By default this tests:

```text
Qwen/Qwen2.5-VL-7B-Instruct
Qwen/Qwen2.5-VL-32B-Instruct
```

on the `flag_core` color set at several image sizes:

```text
red, blue, green, white, black, yellow, orange, light_blue
```

This is the recommended first color battery because it stays close to the flag
game while still separating easy primary/neutral colors from the colors Qwen2.5
struggled with. The results include `color_group_summary.csv`, which groups
colors into primary flag colors, neutral flag colors, and non-primary core flag
colors.

To reproduce the earlier green/orange/purple confusion check:

```bash
COLOR_SET=legacy \
OUT=runs/qwen_visual_legacy_colors \
./scripts/run_qwen_visual_model_sweep.sh
```

To add flag-adjacent variants such as navy, gold, cyan, and teal:

```bash
COLOR_SET=flag_extended \
OUT=runs/qwen_visual_flag_extended \
./scripts/run_qwen_visual_model_sweep.sh
```

To test only the larger Qwen2.5-VL models on an 80 GB GPU:

```bash
MODELS="Qwen/Qwen2.5-VL-32B-Instruct Qwen/Qwen2.5-VL-72B-Instruct" \
OUT=runs/qwen25_visual_32b_72b \
./scripts/run_qwen_visual_model_sweep.sh
```

### LLaVA then Kimi-VL

Use this pair when you want permissive/open-source models outside the Qwen
family. The sweep script defaults to `BACKEND=auto`, so it chooses the loader
from the Hugging Face model id.

If Hugging Face/Xet downloads kill the RunPod web terminal, do not keep trying
the Transformers path for LLaVA. Use the Ollama API path first as a capability
gate. This avoids `snapshot_download` entirely and talks to a local Ollama
server instead.

Install Ollama, keep its model store on `/workspace`, and start the server:

```bash
cd /workspace/flag_game
mkdir -p /workspace/ollama_models runs
export OLLAMA_MODELS=/workspace/ollama_models

curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > runs/ollama_server.log 2>&1 &
```

Pull the smaller Ollama-packaged LLaVA model:

```bash
export OLLAMA_MODELS=/workspace/ollama_models
ollama pull llava:7b
```

If the web terminal is unstable during downloads, launch the pull in the
background and tail the log from a fresh terminal:

```bash
nohup bash -lc 'export OLLAMA_MODELS=/workspace/ollama_models; ollama pull llava:7b' \
  > runs/ollama_pull_llava.log 2>&1 &

tail -f runs/ollama_pull_llava.log
```

Then run the flag-color battery through the Ollama API backend:

```bash
MODELS="llava:7b" \
COLOR_SET=flag_core \
OUT=runs/ollama_llava_visual_flag_core \
./scripts/run_ollama_visual_model_sweep.sh
```

This route is for model-capability screening. It does not give internal
activations. If LLaVA passes the flag-color battery through Ollama, then it is
worth spending effort on a raw-weights/mech-interp setup for the model family.

Run LLaVA first:

```bash
MODELS="llava-hf/llava-v1.6-mistral-7b-hf" \
COLOR_SET=flag_core \
OUT=runs/llava_visual_flag_core \
./scripts/run_qwen_visual_model_sweep.sh
```

For a tiny LLaVA smoke before the full color sweep:

```bash
python scripts/run_qwen_visual_perception_tests.py \
  --model-id llava-hf/llava-v1.6-mistral-7b-hf \
  --suite colors \
  --colors green,orange,light_blue \
  --pixel-sizes 150x100 \
  --max-tests 3 \
  --out runs/llava_visual_smoke
```

Then run Kimi-VL:

```bash
MODELS="moonshotai/Kimi-VL-A3B-Instruct" \
COLOR_SET=flag_core \
OUT=runs/kimi_vl_visual_flag_core \
./scripts/run_qwen_visual_model_sweep.sh
```

Kimi-VL is MIT licensed and modern, but it is much larger on disk than LLaVA:
the Hugging Face repo is about 32.8 GB and uses remote code. The runner enables
`trust_remote_code=True` for Kimi-VL because its model card requires custom
model and processor files.

Compare the model families after both finish:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

runs = [
    Path("runs/qwen_visual_model_sweep/results.csv"),
    Path("runs/llava_visual_flag_core/results.csv"),
    Path("runs/kimi_vl_visual_flag_core/results.csv"),
]
frames = [pd.read_csv(path) for path in runs if path.exists()]
df = pd.concat(frames, ignore_index=True)
summary = (
    df[df["task_type"] == "color"]
    .groupby(["model_id", "expected_color"], dropna=False)
    .agg(
        trials=("trial_id", "count"),
        accuracy=("color_correct", "mean"),
        predictions=("predicted_color", lambda s: sorted(set(map(str, s.dropna())))),
    )
    .reset_index()
)
print(summary.to_string(index=False))
PY
```

For a grouped read of whether a model is only getting primary colors:

```bash
for path in runs/qwen_visual_model_sweep/color_group_summary.csv \
  runs/llava_visual_flag_core/color_group_summary.csv \
  runs/kimi_vl_visual_flag_core/color_group_summary.csv; do
  [ -f "$path" ] && echo "$path" && cat "$path"
done
```

To test the newer Qwen3-VL family, first upgrade Transformers if your current
environment cannot import `Qwen3VLForConditionalGeneration`:

```bash
python -m pip install -U git+https://github.com/huggingface/transformers accelerate
```

Then run:

```bash
MODELS="Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-32B-Instruct" \
OUT=runs/qwen3_visual_8b_32b \
./scripts/run_qwen_visual_model_sweep.sh
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
- `color_group_summary.csv`: flag-color accuracy by primary/neutral/non-primary group
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
