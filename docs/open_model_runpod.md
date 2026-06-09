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
