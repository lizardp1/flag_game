# Open-Model Mech-Interp Pipeline

This is the current path from open-model flag-game runs to representation
geometry analyses. Linear probes are optional later; the first-class question is
how agent representations align, separate, and move across communication rounds.

## Early-Phase Study Plan

The open-model study should start behavior-first, then move to internals only
after the task is demonstrably meaningful for the model family.

### Phase 1: Visible Capabilities And Social Behavior

Run the visual gate first. For LLaVA, the behavior-only Ollama path has already
shown stronger non-primary flag-color recognition than Qwen2.5-VL; use raw
Transformers only when hidden states are needed.

```bash
MODELS="llava:7b" \
COLOR_SET=flag_core \
OUT=runs/ollama_llava_visual_flag_core \
./scripts/run_ollama_visual_model_sweep.sh
```

After you have any mix of visual, flag-game, or memory-conflict outputs, build
a single model-behavior report:

```bash
python scripts/summarize_open_model_behavior.py \
  --visual-run runs/ollama_llava_visual_flag_core \
  --flag-run runs/llava_mech_interp/geometry_batch \
  --memory-conflict-run runs/llava_memory_conflict_probe \
  --out runs/open_model_phase1_behavior
```

If you omit explicit run dirs, the script discovers compatible outputs under
`runs/`.

Key outputs:

```text
runs/open_model_phase1_behavior/model_behavior_report.md
runs/open_model_phase1_behavior/visual_capability_summary.csv
runs/open_model_phase1_behavior/flag_game_behavior_summary.csv
runs/open_model_phase1_behavior/memory_conflict_thresholds.csv
```

For one qualitative LLaVA run that saves a dashboard-style image with the full
flag, crop windows, country trajectory, and outcome summary:

```bash
OUT=runs/llava_visual_report_run \
./scripts/run_llava_visual_report_run.sh
```

Inspect:

```text
runs/llava_visual_report_run/visual_report.html
runs/llava_visual_report_run/plots/run_card.png
```

### Phase 2: Controlled Private-Vs-Social Psychology Probe

The original paper-style phase already lives in:

```text
scripts/run_flag_memory_conflict_probe.py
```

For LLaVA on RunPod, use the wrapper:

```bash
source .venv/bin/activate
source scripts/runpod_cache_env.sh

REPLICATES=3 \
OUT=runs/llava_memory_conflict_probe \
./scripts/run_llava_memory_conflict_probe.sh
```

This tests weak/strong private evidence, compatible/incompatible social
evidence, and every memory composition from `8:0` through `0:8`, using `m=3` by
default.

Key outputs:

```text
runs/llava_memory_conflict_probe/results.csv
runs/llava_memory_conflict_probe/summary_by_count.csv
runs/llava_memory_conflict_probe/threshold_summary.csv
runs/llava_memory_conflict_probe/agent_response_decomposition.png
```

Interpret this before internals. You want to see at least some logically
ordered opinion change as social evidence grows, plus stronger resistance when
private evidence is diagnostic.

### Phase 3: Single-Agent Internal Concept Geometry

Once Phase 2 shows meaningful behavior, rerun the same controlled probe with
compact activation capture:

```bash
ACTIVATION_CAPTURE=1 \
REPLICATES=3 \
OUT=runs/llava_memory_conflict_probe_activations \
./scripts/run_llava_memory_conflict_probe.sh
```

The runner stores `last_prompt_token` and `mean_prompt` for each layer by
default. The wrapper automatically runs:

```bash
python scripts/analyze_memory_conflict_activations.py \
  --runs runs/llava_memory_conflict_probe_activations \
  --feature last_prompt_token \
  --out runs/llava_memory_conflict_probe_activations/activation_concepts_last_prompt_token
```

Key outputs:

```text
runs/llava_memory_conflict_probe_activations/activations/index.jsonl
runs/llava_memory_conflict_probe_activations/activation_concepts_last_prompt_token/activation_samples.csv
runs/llava_memory_conflict_probe_activations/activation_concepts_last_prompt_token/concept_separation_by_layer.csv
runs/llava_memory_conflict_probe_activations/activation_concepts_last_prompt_token/centroid_similarity_by_layer.csv
```

The concept analyzer is not a trained probe. It asks whether concepts cluster in
representation geometry across layers:

- stimulus-side concepts: `private_evidence_strength`, `social_evidence_type`
- social-composition concepts: `false_memory_bin`, `memory_majority`
- decision-side concepts: `response_type`, `choice_axis`, `correct`

This is the early internal-representation bridge before manager/observer
comparisons, polarization cases, or more expensive cross-agent trajectory
analyses.

## 1. Validate The Model

For LLaVA, use the raw Hugging Face/Transformers path when you need
activations. The Ollama backend is useful for behavior-only color screening,
but it cannot expose hidden states for representation geometry.

End-to-end LLaVA pipeline:

```bash
source .venv/bin/activate
source scripts/runpod_cache_env.sh

./scripts/run_llava_mech_interp_pipeline.sh
```

The script runs:

- raw-HF LLaVA visual perception tests on flag colors and stripe patterns
- a tiny pairwise flag-game smoke run with activation capture
- cross-agent/temporal activation geometry analysis
- a multi-seed geometry batch
- optional T=0 linear-probe datasets and probe analyses

To run a faster first pass:

```bash
RUN_BATCH=0 RUN_T0_PROBES=0 RUN_LINEAR_PROBES=0 \
./scripts/run_llava_mech_interp_pipeline.sh
```

The default LLaVA config is:

```text
configs/open_models/llava_1_6_mistral_7b_pairwise_smoke.yaml
```

Run the one-image Qwen smoke first:

```bash
python scripts/run_qwen_vl_smoke.py \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --out runs/qwen_vl_smoke
```

If predictions collapse toward one country, run direct visual tests before
collecting more flag-game data:

```bash
python scripts/run_qwen_visual_perception_tests.py \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --suite all \
  --out runs/qwen_visual_perception_7b
```

Inspect:

```text
runs/qwen_visual_perception_7b/size_summary.csv
runs/qwen_visual_perception_7b/color_summary.csv
runs/qwen_visual_perception_7b/stripe_summary.csv
runs/qwen_visual_perception_7b/breakpoints.csv
```

Only move to representation geometry once the model can reliably name solid
colors and simple stripe patterns at the crop sizes used by the game.

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

## 4. Check Vision And Model Scale Before Overinterpreting Geometry

If behavior collapses toward one answer such as France, start with the
visual-only battery:

```bash
python scripts/run_qwen_visual_perception_tests.py \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --suite all \
  --out runs/qwen_visual_perception_3b_7b
```

This separates raw color/stripe perception from country-choice priors. Then run
the flag-game scale sweep before treating geometry as meaningful social
convergence:

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
