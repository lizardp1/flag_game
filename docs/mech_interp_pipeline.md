# Open-Model Mech-Interp Pipeline

This is the current path from open-model flag-game runs to representation
geometry analyses. Linear probes are optional later; the first-class question is
how agent representations align, separate, and move across communication rounds.

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

## 2. Capture Probe Activations

The Qwen pairwise smoke config enables compact activation capture for every
probe call:

```yaml
activation_capture:
  enabled: true
  scope: all_probes
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

## 3. Compare Representation Geometry

To compare agents across rounds:

```bash
python -m nnd.cli run \
  --config configs/open_models/qwen2_5_vl_7b_pairwise_smoke.yaml \
  --out runs/qwen_geometry_smoke \
  --backend transformers_vlm
```

Then compute cross-agent cosine similarity and temporal drift:

```bash
python scripts/analyze_activation_geometry.py \
  --runs runs/qwen_geometry_smoke \
  --feature last_prompt_token \
  --out runs/qwen_geometry_smoke/activation_geometry
```

This writes:

- `activation_samples.csv`: captured calls and metadata
- `agent_pair_cosine.csv`: cross-agent cosine similarity by `t` and layer
- `agent_temporal_drift.csv`: each agent's drift from its own initial state
- `by_layer_similarity_summary.csv`: average cross-agent similarity by layer
- `by_layer_temporal_summary.csv`: average temporal drift by layer

Useful variants:

```bash
python scripts/analyze_activation_geometry.py \
  --runs runs/qwen_geometry_smoke \
  --feature mean_prompt \
  --out runs/qwen_geometry_smoke/activation_geometry_mean_prompt
```

For a multi-seed geometry dataset:

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

## 4. Check Model Scale Before Overinterpreting Geometry

If behavior collapses toward one answer such as France, run the visual-only
scale sweep before treating the geometry as meaningful social convergence:

```bash
./scripts/run_qwen_scale_capability_sweep.sh
```

This compares Qwen2.5-VL 3B and 7B by default on initial visual probes only
(`T=0`). The summary to inspect is:

```text
runs/qwen_scale_capability/scale_summary/model_scale_metrics.csv
```

The most important diagnostic columns are:

- `initial_agent_accuracy`
- `initial_agent_france_rate`
- `initial_agent_predicted_compatible_rate`
- `france_when_france_incompatible_rate`

Interpretation:

- If larger models reduce France guesses on France-incompatible crops, this is
  probably a capabilities/scale issue.
- If every scale still says France when France is incompatible, the prompt,
  answer schema, crop size, or country pool is likely inducing a prior.
- If oracle accuracy is low, the crop setup itself is too ambiguous for the
  behavioral run to be a clean capability test.

## 5. Optional Linear Probe Dataset

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

## 6. Optional Linear Probes

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

## 7. Remaining Research Steps

After this first slice works:

1. Capture more controlled conditions, especially vision-only vs misleading
   memory.
2. Analyze whether cross-agent similarity increases after communication and
   whether that convergence differs by layer.
3. Add token-region metadata for image tokens, country-list tokens, memory
   tokens, and generated-answer tokens.
4. Add full-sequence activation capture for smaller batches.
5. Optionally train probes for visible colors, stripe orientation, truth country, predicted
   country, correctness, and memory-conflict state.
6. Add causal tests: activation patching from one crop/prompt into another and
   answer-flip measurements.
7. Repeat across more open VLMs once the artifact schema is stable.
