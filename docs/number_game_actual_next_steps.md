# Number Game Actual Next Steps

This is the concrete next-step plan after the local Qwen3-1.7B pre-RunPod checks.

## Local Work

Keep local runs small. The laptop is for validating code paths, prompt formats, plots, and smoke-test behavior.

Already done locally:

- Pairwise `m=1` versus `m=3`, Qwen3-1.7B, 5 seeds, `N=4`.
- Actual pairwise `m=3` trajectory with full dialogue output.
- Corrected flag-style memory-conflict probe, Qwen3-1.7B, 5 seeds, `memory_total=4`.
- Clue information values in bits for every configured clue.
- Conflict-probe phase split: weak, medium, and strong private clues.
- Smoke test for deeper conflict memory: Qwen3-1.7B, `m=3`, 1 seed, `memory_total=8`.

## Clue Information Metric

Each private clue gets a standard self-information value:

```text
I(clue) = -log2 P(clue)
        = log2(total candidate integers / clue-consistent integers)
```

The current behavioral probes prompt the configured integer range, now `1..100`.
The clue-information metric uses an analysis-only uniform baseline over that
range; the model prompt does not say the hidden integer was sampled uniformly.
The model is told this support when `prompt_number_range: true`.

Example: for `1..100`, `the digits sum to 7` matches 8 integers (`7, 16, 25, 34, 43, 52, 61, 70`), so:

```text
I = log2(100 / 8) = 3.64 bits
```

Information phases:

- weak: `I <= 1` bit
- medium: `1 < I <= 2` bits
- strong: `I > 2` bits

In phase plots, `trials=` means the number of model trials contributing to that entire panel. It is not the number of agents.

Local checks still worth running before a cloud sweep:

```bash
python -m unittest tests.test_number_game_memory_format
python scripts/make_number_game_showable_visuals.py
```

Optional local smoke checks:

```bash
HF_HOME=/path/to/hf-cache TRANSFORMERS_OFFLINE=1 \
python -m nnd.number_game.cli conflict-battery \
  --config configs/number_game/local_qwen3_1_7b.yaml \
  --out outputs/qwen3_1_7b_memory_ratio8_m3_1seed_info_smoke \
  --start-seed 0 --num-seeds 1 --ratio-total 8 --m 3
```

Do not run the full model/N/H/seed matrix locally.

## RunPod Behavioral Sweep

Run these on RunPod because they require enough calls that local iteration becomes slow.

### 1. Social/private conflict probe

Purpose: reproduce the flag-game Figure-6-style result with number clues.

Run for each model:

```bash
python -m nnd.number_game.cli conflict-battery \
  --config configs/number_game/runpod_qwen3.yaml \
  --out OUT_DIR/memory_ratio8 \
  --start-seed 0 --num-seeds 5 \
  --ratio-total 8 \
  --m 1 --m 3 \
  --override min_number=1 \
  --override max_number=100 \
  --override prompt_number_range=true \
  --override prompt_social_susceptibility=false
```

Primary outputs:

- `conflict_summary.csv`
- `conflict_phase_summary.csv`
- `clue_information.csv`
- `conflict_trials.csv`

Main plot:

- response composition over `8:0 ... 0:8`
- split by private-clue information phase
- rows for `m=1` and `m=3`

### 2. Actual social interaction sweep

Purpose: show that social interactions actually change outcomes, not only single-agent probes.

Run pairwise first:

```bash
python -m nnd.number_game.cli compare-pairwise-m \
  --config configs/number_game/runpod_qwen3.yaml \
  --out OUT_DIR/pairwise_m_compare \
  --start-seed 0 --num-seeds 5 \
  --m 1 --m 3 \
  --override min_number=1 \
  --override max_number=100 \
  --override prompt_number_range=true \
  --override early_stop_window=5 \
  --override consensus_threshold=0.9
```

Sweep:

- `N = 4, 8, 16, 32`
- `H = 1, 3, 5, 8`
- `m = 1, 3`
- `probe_every = max(N // 2, 1)`
- 5 seeds to start

Required outputs per condition:

- number-share trajectory plot
- dialogue transcript for `m=3`
- probe-change table
- changed-to-truth and changed-away-from-truth counts
- final consensus correctness
- private clue satisfaction

### 3. Broadcast and org protocols

Purpose: establish parity with the base flag-game protocol families.

Run after pairwise is stable:

- `broadcast`, `m=3`, `N=8,16`, `H=8`, 5 seeds
- `org`, `m=3`, `N=8,16`, `H=8`, 5 seeds

## RunPod Model Set

Minimum first pass:

- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`
- Kimi through an OpenAI-compatible endpoint if hosted successfully

Second pass:

- larger Qwen model
- larger Kimi model
- optional Llama baseline

Keep exact prompts, seeds, memory ratios, and clue information bins fixed across models.

## Mechanistic Interpretability Sweep

Current status:

- [x] Auto-generated synthetic contrast cases with train/test split.
- [x] Layerwise mean social-minus-private activation directions.
- [x] Alpha sweep over full-sequence number log probabilities.
- [x] Empirical sign calibration, because the raw contrast-vector sign can flip under intervention.
- [x] Sign-calibrated vector file where positive calibrated alpha means empirically more social.
- [x] Full literature-style steering evaluation suite implemented in `scripts/run_number_game_steering_prep.py`.
- [ ] Full suite run on RunPod for Qwen3-8B and larger/Kimi models.

### Steering Figure and Evaluation Backlog

These are implemented outputs, but the expensive ones still need to be run on
RunPod before treating the steering vector as a serious result:

- [x] Projection distributions: social/private projection histograms and scatter plots, split by train/test.
- [x] Layer x alpha heatmap: effect size over layers and steering strengths.
- [x] Data-scaling curve: fit directions with 8, 16, 32, 64, and 128 cases to check sample-size effects and saturation.
- [x] Full-sequence number log probability, replacing the old first-token number-logit approximation.
- [x] Generation-time steering: actual JSON outputs, final choices, clue satisfaction, and parser validity under alpha.
- [x] Side-effect charts: validity, perplexity, and format damage versus alpha.
- [x] Vector stability: cosine similarity across random seeds and case subsets.
- [x] Qualitative examples and `m=3` dialogues at representative alpha values.
- [x] OOD/generalization: train on synthetic probes, test inside actual social-game prompts.

Interpretation rule: the raw vector is a diagnostic contrast. Causal claims should
use the empirically calibrated sign and should be backed by generation-time
outputs, not just answer-token logits.

### Literature Alignment

The steering suite follows the current activation-steering pattern:

- ActAdd-style contrast vectors: compute hidden-state differences between paired prompts and add the vector at inference time.
- CAA-style evaluation: sweep layers and multipliers, test held-out prompts, inspect open-ended generations, check side effects, and compare vector similarities.
- CAE practical checks: run data-scaling curves, because reported returns diminish around roughly 80 contrast examples; test OOD prompts separately from the synthetic vector-building distribution; track perplexity/format damage.
- Geometry checks: use projection distributions and train/test separability, matching the broader representational-geometry practice of asking whether latent classes form stable, separated regions rather than trusting a single aggregate score.

Useful references:

- Activation Addition: https://arxiv.org/abs/2308.10248
- Contrastive Activation Addition: https://arxiv.org/abs/2312.06681
- Patterns and Mechanisms of Contrastive Activation Engineering: https://openreview.net/forum?id=FZk9oWvZm2
- Latent Structure of Affective Representations in LLMs: https://arxiv.org/abs/2604.07382

### 1. Contrast-pair dataset

Create paired prompts where only the evidence ownership/source changes:

- private clue supports A, memory supports B
- private clue supports B, memory supports A
- compatible social memory
- contradictory social memory
- decisive social memory
- weak, medium, and strong private-clue phases

### 2. Activation capture

Current local code captures last-token and mean prompt hidden states. The next version should capture position-specific residual streams:

- private clue tokens
- memory number tokens
- memory reason tokens
- final instruction token
- pre-answer token

### 3. Linear probes

Train and evaluate probes by layer and token position:

- private target number
- social memory number
- final chosen number
- response category: private, social, other clue-compatible, other clue-incompatible

Use held-out seeds, held-out clue cases, and held-out memory ratios.

### 4. Steering vectors

Derive candidate vectors from:

- mean activation contrast: private-wins minus social-wins
- linear classifier normal vector

Test both signs:

- private-up steering
- social-up steering

### 5. Causal steering evaluation

Apply steering during generation and measure:

- full-sequence number-logprob shifts
- final answer flips
- JSON validity
- private clue satisfaction
- rationale coherence under `m=3`
- social trajectory effects in actual pairwise runs

### 6. Layer localization

Expected analysis:

- early layers: reading/parsing clues and memory
- middle layers: integrating private and social evidence
- late layers: answer selection/logit commitment

Validate with probe accuracy by layer, activation patching, and steering dose-response by layer.
