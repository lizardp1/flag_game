# Open-Model Mech-Interp Pipeline

This is the current path from open-model flag-game runs to first mechanistic
interpretability analyses.

## 1. Validate The Model

Run the one-image Qwen smoke first:

```bash
python scripts/run_qwen_vl_smoke.py \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --out runs/qwen_vl_smoke
```

Then run the tiny pairwise game:

```bash
python -m nnd.cli run \
  --config configs/open_models/qwen2_5_vl_7b_pairwise_smoke.yaml \
  --out runs/qwen_pairwise_smoke \
  --backend transformers_vlm
```

## 2. Capture Initial-Probe Activations

The Qwen pairwise smoke config enables compact activation capture:

```yaml
activation_capture:
  enabled: true
  scope: initial_probe
  save_full_sequence: false
  storage_dtype: float16
```

For each captured call, the runner writes:

```text
activations/index.jsonl
activations/tensors/<call_id>.pt
```

Each tensor shard contains:

- `last_prompt_token`: layer-by-hidden tensor for the final prompt token
- `mean_prompt`: layer-by-hidden tensor averaged over prompt tokens
- `layers`, `input_ids`, `attention_mask`, `tokens`
- call metadata linking back to `probes.jsonl` and `trial_manifest.json`

## 3. Build A Small Probe Dataset

Run initial probes across multiple seeds:

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

This gives `2 * num_seeds` activation samples because the smoke config uses
`N=2`.

## 4. Train First Linear Probes

Truth-country probe:

```bash
python scripts/analyze_activation_linear_probe.py \
  --runs runs/qwen_activation_t0_batch \
  --target truth_country \
  --feature last_prompt_token \
  --out runs/qwen_activation_t0_batch/linear_probe_truth_country
```

Crop-informativeness probe:

```bash
python scripts/analyze_activation_linear_probe.py \
  --runs runs/qwen_activation_t0_batch \
  --target informativeness_label \
  --feature mean_prompt \
  --out runs/qwen_activation_t0_batch/linear_probe_informativeness
```

Supported targets are:

```text
truth_country
predicted_country
correct
truth_compatible
is_unique
informativeness_label
compatible_country_count
```

## 5. Remaining Research Steps

After this first slice works:

1. Capture more controlled conditions, especially vision-only vs misleading
   memory.
2. Add token-region metadata for image tokens, country-list tokens, memory
   tokens, and generated-answer tokens.
3. Add full-sequence activation capture for smaller batches.
4. Train probes for visible colors, stripe orientation, truth country, predicted
   country, correctness, and memory-conflict state.
5. Add causal tests: activation patching from one crop/prompt into another and
   answer-flip measurements.
6. Repeat across more open VLMs once the artifact schema is stable.
