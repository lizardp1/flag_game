# Number Game Rerun Checklist

The prompt contract changed. Treat earlier number-game outputs as pre-change
smoke tests unless their resolved config says:

```yaml
min_number: 1
max_number: 100
prompt_number_range: true
prompt_social_susceptibility: false
```

The new prompt explicitly tells agents:

```text
The hidden integer was sampled uniformly from the integers 1 through 100, inclusive.
```

This makes the information-theory clue metric interpretable as the model-visible
uniform prior:

```text
I(clue) = -log2 P(clue)
        = log2(100 / clue-consistent-count)
```

## Status

- [x] Code supports `prompt_number_range`.
- [x] Active configs reduced to three handoff-safe files.
- [x] Active configs use prompted `1..100`.
- [x] Conflict rows include clue candidate counts, prior probability, bits, and phase.
- [x] Visual code labels clue information as self-information and uses the actual candidate count denominator.
- [x] Rerun local Qwen3-1.7B smoke results under the new prompt.
- [ ] Rerun RunPod Qwen behavioral matrix under the new prompt.
- [ ] Rerun Kimi endpoint behavioral matrix under the new prompt.
- [ ] Regenerate all showable figures from new prompted-prior outputs.
- [ ] Start activation/probe/steering runs only after the behavioral reruns pass sanity checks.

## Files To Use

- `configs/number_game/local_qwen3_1_7b.yaml`: local Qwen3-1.7B smoke runs.
- `configs/number_game/runpod_qwen3.yaml`: RunPod Qwen template.
- `configs/number_game/runpod_kimi_k2_endpoint.yaml`: Kimi K2 OpenAI-compatible endpoint template.

Use CLI overrides for model size, protocol, population size, memory size, and
hidden-state capture.

## Preflight Checks

- [x] Run unit tests.

```bash
python -m unittest tests.test_number_game_memory_format
```

- [x] Compile number-game modules and scripts.

```bash
python -m py_compile \
  nnd/number_game/config.py \
  nnd/number_game/prompts.py \
  nnd/number_game/backend.py \
  nnd/number_game/runner.py \
  nnd/number_game/conflict.py \
  nnd/number_game/cli.py \
  scripts/run_number_game_pre_runpod_probes.py \
  scripts/make_number_game_showable_visuals.py
```

- [x] Verify all active configs load.

```bash
python -c 'from pathlib import Path; from nnd.number_game.config import load_number_game_config; [load_number_game_config(p) for p in Path("configs/number_game").glob("*.yaml") if p.name != "README.md"]; print("ok")'
```

- [x] Save or inspect one exact prompt sample and confirm it has the range sentence but no allowed-number list.

```bash
python -c 'from nnd.number_game import prompts; print(prompts.interaction_text(numbers=list(range(1,101)), private_clue="the number is odd", memory_lines=["12", "7 | The number is prime."], m=3, prompt_social_susceptibility=False, prompt_number_range=True))'
```

## Local Reruns

Keep local runs small. The goal is to prove the new prompt and metrics work, not
to make final claims.

- [x] Pairwise `m=1` vs `m=3`, Qwen3-1.7B, 5 seeds, `N=4`.

```bash
python -m nnd.number_game.cli compare-pairwise-m \
  --config configs/number_game/local_qwen3_1_7b.yaml \
  --out outputs/number_game_prompt100/local_qwen3_1_7b_pairwise_m_compare_5seeds \
  --start-seed 0 \
  --num-seeds 5 \
  --m 1 \
  --m 3
```

- [x] Actual pairwise `m=3` trajectory with dialogues.

```bash
python -m nnd.number_game.cli run \
  --config configs/number_game/local_qwen3_1_7b.yaml \
  --out outputs/number_game_prompt100/local_qwen3_1_7b_pairwise_m3_seed0_trajectory \
  --seed 0 \
  --override interaction_m=3 \
  --override T=80 \
  --override H=8
```

- [x] Social/private conflict probe, local smoke, `memory_total=8`.

```bash
python -m nnd.number_game.cli conflict-battery \
  --config configs/number_game/local_qwen3_1_7b.yaml \
  --out outputs/number_game_prompt100/local_qwen3_1_7b_conflict_ratio8_1seed \
  --start-seed 0 \
  --num-seeds 1 \
  --ratio-total 8 \
  --m 1 \
  --m 3
```

Local success criteria:

- [x] `config_resolved.yaml` shows `max_number: 100` and `prompt_number_range: true`.
- [x] Prompt diagnostics show the exact range sentence.
- [x] `clue_information.csv` has `candidate_range_count=100`.
- [x] Conflict outputs include weak/medium/strong clue phases.
- [x] Pairwise outputs include `dialogues.md`, probe timelines, and number-share trajectory plots.

## Latest Local Rerun Results

Completed on 2026-07-23 with `Qwen/Qwen3-1.7B`.

Outputs:

- `outputs/number_game_prompt100/local_qwen3_1_7b_pairwise_m_compare_5seeds`
- `outputs/number_game_prompt100/local_qwen3_1_7b_pairwise_m3_seed0_trajectory`
- `outputs/number_game_prompt100/local_qwen3_1_7b_conflict_ratio8_1seed`

Preflight:

- unit tests: 7 passed
- py_compile: passed
- active configs loaded: 3
- exact prompt sample included the `1 through 100` range sentence and no allowed-number list

Pairwise `m=1` vs `m=3`, 5 seeds:

- `m=1`: final accuracy `0.20`, consensus-correct `0.20`, valid rate `1.00`
- `m=3`: final accuracy `0.20`, consensus-correct `0.20`, valid rate `1.00`

Dedicated `m=3` trajectory, seed 0:

- truth number: `50`
- final consensus: `50`
- final accuracy: `1.00`
- early stopped at `t=20` after 5 consensus probe rounds
- changed-to-truth count: `2`
- changed-away-from-truth count: `0`

Conflict smoke, 1 seed, `memory_total=8`:

- prompt rows: 108/108 contain the prompted `1..100` prior
- `clue_information.csv` uses `candidate_range_count=100`
- all conflict rows have valid JSON outputs
- at `4:4` memory, private target rate was `0.83` for `m=1` and `0.67` for `m=3`
- at `2:6` memory, private target rate was `0.50` for `m=1` and `0.17` for `m=3`
- at `0:8` memory, social-evidence rate was `0.67` for `m=1` and `1.00` for `m=3`

## RunPod Behavioral Reruns

These are the first results worth comparing across models.

- [ ] Qwen3-1.7B conflict probe, 5 seeds, `memory_total=8`.
- [ ] Qwen3-4B conflict probe, 5 seeds, `memory_total=8`.
- [ ] Qwen3-8B conflict probe, 5 seeds, `memory_total=8`.

```bash
python -m nnd.number_game.cli conflict-battery \
  --config configs/number_game/runpod_qwen3.yaml \
  --out outputs/number_game_prompt100/MODEL_SLUG_conflict_ratio8_5seeds \
  --start-seed 0 \
  --num-seeds 5 \
  --ratio-total 8 \
  --m 1 \
  --m 3 \
  --override model=MODEL_ID
```

- [ ] Qwen3-1.7B pairwise actual social interaction sweep.
- [ ] Qwen3-4B pairwise actual social interaction sweep.
- [ ] Qwen3-8B pairwise actual social interaction sweep.

Start with:

```bash
python -m nnd.number_game.cli compare-pairwise-m \
  --config configs/number_game/runpod_qwen3.yaml \
  --out outputs/number_game_prompt100/MODEL_SLUG_pairwise_m_compare_5seeds \
  --start-seed 0 \
  --num-seeds 5 \
  --m 1 \
  --m 3 \
  --override model=MODEL_ID \
  --override N=8 \
  --override H=8 \
  --override T=160
```

Then sweep:

- [ ] `N = 4, 8, 16, 32`
- [ ] `H = 1, 3, 5, 8`
- [ ] `m = 1, 3`
- [ ] 5 seeds per condition
- [ ] keep `early_stop_window=5`

## Kimi Reruns

Run Kimi through the OpenAI-compatible endpoint config.

- [ ] Kimi K2 conflict probe, 5 seeds, `memory_total=8`.

```bash
python -m nnd.number_game.cli conflict-battery \
  --config configs/number_game/runpod_kimi_k2_endpoint.yaml \
  --out outputs/number_game_prompt100/kimi_k2_conflict_ratio8_5seeds \
  --start-seed 0 \
  --num-seeds 5 \
  --ratio-total 8 \
  --m 1 \
  --m 3
```

- [ ] Kimi K2 pairwise actual social interaction run, 5 seeds.

```bash
python -m nnd.number_game.cli compare-pairwise-m \
  --config configs/number_game/runpod_kimi_k2_endpoint.yaml \
  --out outputs/number_game_prompt100/kimi_k2_pairwise_m_compare_5seeds \
  --start-seed 0 \
  --num-seeds 5 \
  --m 1 \
  --m 3
```

## Figures To Regenerate

Regenerate figures only from prompted `1..100` outputs.

- [ ] Clue information chart with `100` as the denominator.
- [ ] Flag-style social/private conflict probe over memory ratios.
- [ ] Conflict probe split by clue information phase.
- [ ] Actual pairwise number-share trajectories for every model/condition.
- [ ] Dialogue excerpts for `m=3` showing rationales.
- [ ] Summary table: validity, private-clue satisfaction, final accuracy, consensus, changed-to-truth, changed-away-from-truth.

## Mechanistic Interpretability Gate

Do not run large activation sweeps until the behavioral reruns look sane.

- [ ] Build contrast-pair prompts under the same prompted `1..100` contract.
- [ ] Capture hidden states for private clue tokens, memory number tokens, memory reason tokens, final instruction token, and pre-answer token.
- [ ] Train/test linear probes for private target, social target, final choice, and response category.
- [ ] Derive private-vs-social steering vectors from contrast-pair means and/or probe normals.
- [ ] Apply activation steering during generation.
- [ ] Measure answer-logit shifts, final-choice shifts, JSON validity, clue satisfaction, and rationale coherence.
- [ ] Localize layers into reading, integration, and answer-selection regions.

## Stale Outputs

Do not cite old output folders whose resolved config has `max_number: 30`,
`max_number: 12`, or `prompt_number_range: false`.

Those folders remain useful only for checking that code paths and plotting
scripts once worked.
