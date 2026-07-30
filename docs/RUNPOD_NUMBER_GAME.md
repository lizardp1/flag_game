# RunPod Number Game Instructions

Use this after the local prompted-range checks have passed. The current
experiment contract is:

```yaml
min_number: 1
max_number: 100
prompt_number_range: true
prompt_social_susceptibility: false
early_stop_window: 5
```

The model is explicitly told only the integer range:

```text
The hidden integer is in the range 1 through 100, inclusive.
```

Old number-game outputs are stale unless their `config_resolved.yaml` matches
that contract and their saved prompts contain the range-only sentence above.

## 1. Pod Setup

Recommended starting point:

- GPU: one high-memory GPU for Qwen3-8B; smaller Qwen models can run on less.
- Disk: at least 100 GB if testing multiple model sizes.
- Image: PyTorch image with Python 3.10+.

From the pod:

```bash
git clone https://github.com/lizardp1/flag_game.git
cd flag_game
git checkout number-game-open-models

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers accelerate numpy matplotlib
```

If the pod has CUDA-specific PyTorch requirements, install the matching PyTorch
wheel first, then install the remaining packages.

## 2. Preflight

Run these before spending GPU time on sweeps:

```bash
python -m unittest tests.test_number_game_memory_format

python -m py_compile \
  nnd/number_game/config.py \
  nnd/number_game/prompts.py \
  nnd/number_game/backend.py \
  nnd/number_game/runner.py \
  nnd/number_game/conflict.py \
  nnd/number_game/cli.py \
  scripts/run_number_game_pre_runpod_probes.py \
  scripts/make_number_game_showable_visuals.py

python -c 'from pathlib import Path; from nnd.number_game.config import load_number_game_config; [load_number_game_config(p) for p in Path("configs/number_game").glob("*.yaml") if p.name != "README.md"]; print("configs ok")'
```

Verify the exact prompt contract:

```bash
python -c 'from nnd.number_game import prompts; print(prompts.interaction_text(numbers=list(range(1,101)), private_clue="the number is odd", memory_lines=["12 | The number is even.", "7 | The number is prime."], m=3, prompt_social_susceptibility=False, prompt_number_range=True))'
```

The printed prompt should include `1 through 100` and should not include an
`Allowed numbers` list.

## 3. Qwen Conflict Probe

Run the private-vs-social conflict probe first. This is the closest number-game
analogue of the flag-game memory-conflict figure.

```bash
MODEL_ID=Qwen/Qwen3-1.7B
MODEL_SLUG=qwen3_1_7b

python -m nnd.number_game.cli conflict-battery \
  --config configs/number_game/runpod_qwen3.yaml \
  --out outputs/number_game_prompt100/${MODEL_SLUG}_conflict_ratio8_5seeds \
  --start-seed 0 \
  --num-seeds 5 \
  --ratio-total 8 \
  --m 1 \
  --m 3 \
  --override model=${MODEL_ID}
```

Repeat with:

```bash
MODEL_ID=Qwen/Qwen3-4B
MODEL_SLUG=qwen3_4b
```

and:

```bash
MODEL_ID=Qwen/Qwen3-8B
MODEL_SLUG=qwen3_8b
```

Sanity checks:

- `config_resolved.yaml` has `max_number: 100` and `prompt_number_range: True`.
- `clue_information.csv` has `candidate_range_count=100`.
- `conflict_prompts.jsonl` contains the `1 through 100` range sentence.
- `conflict_phase_summary.csv` has weak, medium, and strong clue rows.
- `valid_rate` is close to 1.0; if not, inspect `debug/`.

## 4. Qwen Actual Pairwise Social Runs

Start with one model and `N=8`, `H=8`, 5 seeds:

```bash
MODEL_ID=Qwen/Qwen3-1.7B
MODEL_SLUG=qwen3_1_7b

python -m nnd.number_game.cli compare-pairwise-m \
  --config configs/number_game/runpod_qwen3.yaml \
  --out outputs/number_game_prompt100/${MODEL_SLUG}_pairwise_N8_H8_5seeds \
  --start-seed 0 \
  --num-seeds 5 \
  --m 1 \
  --m 3 \
  --override model=${MODEL_ID} \
  --override N=8 \
  --override H=8 \
  --override T=160 \
  --override probe_every=8
```

Then sweep:

- `N = 4, 8, 16, 32`
- `H = 1, 3, 5, 8`
- `m = 1, 3`
- 5 seeds

For each condition, set `probe_every` to about `N`.

## 5. Broadcast And Org Parity

After pairwise is sane:

```bash
python -m nnd.number_game.cli run \
  --config configs/number_game/runpod_qwen3.yaml \
  --out outputs/number_game_prompt100/qwen3_4b_broadcast_seed0 \
  --seed 0 \
  --protocol broadcast \
  --override model=Qwen/Qwen3-4B \
  --override interaction_m=3 \
  --override N=8 \
  --override H=8
```

```bash
python -m nnd.number_game.cli run \
  --config configs/number_game/runpod_qwen3.yaml \
  --out outputs/number_game_prompt100/qwen3_4b_org_seed0 \
  --seed 0 \
  --protocol org \
  --override model=Qwen/Qwen3-4B \
  --override interaction_m=3 \
  --override N=8 \
  --override H=8
```

## 6. Kimi Endpoint

Kimi K2 is expected to run behind an OpenAI-compatible server such as vLLM or
SGLang. Start the server separately, then:

```bash
export NND_MODEL_BASE_URL=http://localhost:8000/v1
export NND_MODEL_API_KEY=local
```

Conflict probe:

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

Pairwise social interaction:

```bash
python -m nnd.number_game.cli compare-pairwise-m \
  --config configs/number_game/runpod_kimi_k2_endpoint.yaml \
  --out outputs/number_game_prompt100/kimi_k2_pairwise_N8_H8_5seeds \
  --start-seed 0 \
  --num-seeds 5 \
  --m 1 \
  --m 3
```

This code uses plain HTTP against `/v1/chat/completions`; it does not require
the OpenAI Python client.

## 7. Figures

After conflict and pairwise outputs exist:

```bash
python scripts/make_number_game_showable_visuals.py \
  --pairwise-dir outputs/number_game_prompt100/qwen3_1_7b_pairwise_N8_H8_5seeds \
  --conflict-dir outputs/number_game_prompt100/qwen3_1_7b_conflict_ratio8_5seeds \
  --out-dir outputs/number_game_prompt100/qwen3_1_7b_showable_visuals
```

Expected files:

- `00_clue_information_values.svg`
- `01_actual_trajectories_pairwise_m1_5seeds.svg`
- `02_actual_trajectories_pairwise_m3_5seeds.svg`
- `03_private_vs_social_memory_ratio_flag_style.svg`
- `04_conflict_probe_by_clue_information_phase.svg`

## 8. Mech-Interp Gate

Do not start large activation sweeps until the behavioral outputs pass sanity
checks. The first local/RunPod steering-prep pass is:

```bash
python scripts/run_number_game_steering_prep.py \
  --config configs/number_game/runpod_qwen3.yaml \
  --out outputs/number_game_prompt100/qwen3_8b_steering_prep \
  --memory-strength 1 \
  --memory-strength 4 \
  --memory-strength 8 \
  --m 1 \
  --m 3 \
  --layer 8 \
  --layer 16 \
  --layer 24 \
  --layer 28 \
  --alpha -2 \
  --alpha -1 \
  --alpha -0.5 \
  --alpha 0 \
  --alpha 0.5 \
  --alpha 1 \
  --alpha 2 \
  --override model=Qwen/Qwen3-8B \
  --override trust_remote_code=false
```

For a cheap local Qwen3-1.7B direction smoke, skip the alpha sweep:

```bash
python scripts/run_number_game_steering_prep.py \
  --config configs/number_game/local_qwen3_1_7b.yaml \
  --out outputs/number_game_steering_prep/qwen3_1_7b_local_all_cases_direction \
  --memory-strength 4 \
  --m 1 \
  --m 3 \
  --layer 8 \
  --layer 16 \
  --layer 24 \
  --layer 28 \
  --skip-alpha-sweep \
  --override trust_remote_code=false
```

This writes `steering_vectors_social_minus_private.npz`,
`steering_direction_summary.csv`, and `steering_alpha_sweep.csv`. Positive alpha
adds the social-memory direction; negative alpha pushes back toward private or
target-memory evidence. After this pass, train/test linear probes by layer and
token position, then test whether generation-time steering shifts final choices
without damaging JSON validity or clue satisfaction.
