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

The current prompted-prior behavioral probes use a uniform prior over the configured integer range, now `1..100`.
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

- number-logit shifts
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
