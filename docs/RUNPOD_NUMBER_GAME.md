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
`Allowed numbers` list. Transcript memory entries should be bare lines, e.g.
`12 | The number is even.`, not hyphen bullets like `- 12`, because numeric
bullets can look like negative numbers.

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
checks. The four handwritten contrast cases are only a smoke test. First use the
auto-generated train/test battery for actual layer selection:

```bash
python scripts/run_number_game_steering_prep.py \
  --config configs/number_game/runpod_qwen3.yaml \
  --out outputs/number_game_prompt100/qwen3_8b_steering_prep_auto64_direction \
  --case-source auto \
  --max-cases 64 \
  --targets-per-clue 4 \
  --socials-per-target 2 \
  --case-seed 0 \
  --test-frac 0.25 \
  --fit-split train \
  --memory-strength 1 \
  --memory-strength 4 \
  --memory-strength 8 \
  --m 1 \
  --m 3 \
  --layer 8 \
  --layer 16 \
  --layer 24 \
  --layer 28 \
  --skip-alpha-sweep \
  --override model=Qwen/Qwen3-8B \
  --override trust_remote_code=false
```

This fits the social-minus-private direction on train cases and uses held-out
test cases for the layer-summary plot when test rows exist. It also writes the
cheap diagnostics that do not require generation: projection distributions,
data-scaling rows, and vector-stability rows. Then run a capped alpha check on
the best late layers:

```bash
python scripts/run_number_game_steering_prep.py \
  --config configs/number_game/runpod_qwen3.yaml \
  --out outputs/number_game_prompt100/qwen3_8b_steering_prep_auto64_alpha \
  --case-source auto \
  --max-cases 64 \
  --targets-per-clue 4 \
  --socials-per-target 2 \
  --case-seed 0 \
  --test-frac 0.25 \
  --fit-split train \
  --memory-strength 1 \
  --memory-strength 4 \
  --memory-strength 8 \
  --m 1 \
  --m 3 \
  --layer 24 \
  --layer 28 \
  --data-scaling-size 8 \
  --data-scaling-size 16 \
  --data-scaling-size 32 \
  --data-scaling-size 64 \
  --data-scaling-size 128 \
  --alpha -10 \
  --alpha -5 \
  --alpha -2 \
  --alpha -1 \
  --alpha 0 \
  --alpha 1 \
  --alpha 2 \
  --alpha 5 \
  --alpha 10 \
  --max-alpha-trials 192 \
  --override model=Qwen/Qwen3-8B \
  --override trust_remote_code=false
```

For the full steering-evaluation pass, include generation-time steering. This
is the pass that writes actual JSON outputs, final choices, clue satisfaction,
validity, format damage, and completion-perplexity side-effect charts:

```bash
python scripts/run_number_game_steering_prep.py \
  --config configs/number_game/runpod_qwen3.yaml \
  --out outputs/number_game_prompt100/qwen3_8b_steering_eval_auto176_generation \
  --case-source auto \
  --max-cases 176 \
  --targets-per-clue 8 \
  --socials-per-target 2 \
  --case-seed 0 \
  --test-frac 0.25 \
  --fit-split train \
  --memory-strength 1 \
  --memory-strength 4 \
  --memory-strength 8 \
  --m 1 \
  --m 3 \
  --layer 24 \
  --layer 28 \
  --data-scaling-size 8 \
  --data-scaling-size 16 \
  --data-scaling-size 32 \
  --data-scaling-size 64 \
  --data-scaling-size 128 \
  --alpha -10 \
  --alpha -5 \
  --alpha -2 \
  --alpha -1 \
  --alpha 0 \
  --alpha 1 \
  --alpha 2 \
  --alpha 5 \
  --alpha 10 \
  --max-alpha-trials 256 \
  --run-generation-steering \
  --max-generation-trials 96 \
  --qualitative-examples-per-alpha 3 \
  --override model=Qwen/Qwen3-8B \
  --override trust_remote_code=false
```

After you have actual pairwise social-game outputs, run the OOD generalization
check by pointing `--ood-social-dir` at the pairwise output root. This trains
the vector on synthetic conflict probes and tests it inside reconstructed real
pairwise prompts:

```bash
python scripts/run_number_game_steering_prep.py \
  --config configs/number_game/runpod_qwen3.yaml \
  --out outputs/number_game_prompt100/qwen3_8b_steering_eval_auto176_ood \
  --case-source auto \
  --max-cases 176 \
  --targets-per-clue 8 \
  --socials-per-target 2 \
  --case-seed 0 \
  --test-frac 0.25 \
  --fit-split train \
  --memory-strength 1 \
  --memory-strength 4 \
  --memory-strength 8 \
  --m 1 \
  --m 3 \
  --layer 24 \
  --layer 28 \
  --alpha -10 \
  --alpha -5 \
  --alpha -2 \
  --alpha -1 \
  --alpha 0 \
  --alpha 1 \
  --alpha 2 \
  --alpha 5 \
  --alpha 10 \
  --max-alpha-trials 256 \
  --run-generation-steering \
  --max-generation-trials 96 \
  --ood-social-dir outputs/number_game_prompt100/qwen3_8b_pairwise_N8_H8_5seeds \
  --max-ood-prompts 128 \
  --override model=Qwen/Qwen3-8B \
  --override trust_remote_code=false
```

For a cheap local Qwen3-1.7B direction smoke, skip the alpha sweep:

```bash
python scripts/run_number_game_steering_prep.py \
  --config configs/number_game/local_qwen3_1_7b.yaml \
  --out outputs/number_game_steering_prep/qwen3_1_7b_auto_cases_tiny \
  --case-source auto \
  --max-cases 4 \
  --targets-per-clue 4 \
  --socials-per-target 2 \
  --case-seed 0 \
  --test-frac 0.25 \
  --fit-split train \
  --memory-strength 4 \
  --m 1 \
  --layer 16 \
  --layer 28 \
  --skip-alpha-sweep \
  --override trust_remote_code=false
```

This writes `steering_vectors_social_minus_private.npz`,
`steering_cases.csv`, `steering_direction_summary.csv`,
`projection_distributions.csv`, `data_scaling_curve.csv`,
`vector_stability.csv`, and `steering_alpha_sweep.csv`. If the alpha sweep
runs, it also writes `steering_sign_summary.csv`,
`steering_vectors_empirical_social.npz`, `layer_alpha_heatmap.svg`, and
calibrated alpha columns. If generation steering runs, it writes
`generation_steering_outputs.jsonl`, `generation_side_effect_summary.csv`,
`qualitative_alpha_examples.md`, and the matching side-effect plots. If
`--ood-social-dir` is supplied, it also writes `ood_social_*` generalization
files.

The raw vector is diagnostic: `social_memory_activation - private_memory_activation`.
Do not assume its raw sign is causal. Use `calibrated_alpha` in
`steering_alpha_sweep.csv` and `steering_vectors_empirical_social.npz` for
causal steering, where positive calibrated alpha means empirically more social.
The alpha sweep now scores full-sequence number log probabilities, not just the
first token after `{"number":`.

### Follow-Up: Stronger Causal Checks

The first full run showed a real behavioral shift, but social evidence did not
fully win in logprob space. Use this broader layer/alpha run to check whether
there is a better causal layer and whether stronger steering flips choices
before side effects become unacceptable:

```bash
python scripts/run_number_game_steering_prep.py \
  --config configs/number_game/runpod_qwen3.yaml \
  --out outputs/number_game_prompt100/qwen3_8b_steering_eval_memory_contrast_broad_alpha \
  --case-source auto \
  --max-cases 176 \
  --targets-per-clue 8 \
  --socials-per-target 2 \
  --case-seed 0 \
  --test-frac 0.25 \
  --fit-split train \
  --direction-method memory_contrast \
  --memory-strength 1 \
  --memory-strength 4 \
  --memory-strength 8 \
  --m 1 \
  --m 3 \
  --layer 16 \
  --layer 18 \
  --layer 20 \
  --layer 22 \
  --layer 24 \
  --layer 26 \
  --layer 28 \
  --alpha -30 \
  --alpha -20 \
  --alpha -15 \
  --alpha -10 \
  --alpha -5 \
  --alpha 0 \
  --alpha 5 \
  --alpha 10 \
  --alpha 15 \
  --alpha 20 \
  --alpha 30 \
  --max-alpha-trials 256 \
  --run-generation-steering \
  --max-generation-trials 96 \
  --qualitative-examples-per-alpha 3 \
  --override model=Qwen/Qwen3-8B \
  --override trust_remote_code=false
```

Then test outcome-aligned and subspace alternatives. These are the checks for
whether the original vector was mostly an evidence-source vector rather than a
choice-control vector:

```bash
python scripts/run_number_game_steering_prep.py \
  --config configs/number_game/runpod_qwen3.yaml \
  --out outputs/number_game_prompt100/qwen3_8b_steering_eval_logprob_quantile \
  --case-source auto \
  --max-cases 176 \
  --targets-per-clue 8 \
  --socials-per-target 2 \
  --case-seed 0 \
  --test-frac 0.25 \
  --fit-split train \
  --direction-method logprob_quantile \
  --direction-quantile 0.25 \
  --memory-strength 1 \
  --memory-strength 4 \
  --memory-strength 8 \
  --m 1 \
  --m 3 \
  --layer 20 \
  --layer 22 \
  --layer 24 \
  --layer 26 \
  --layer 28 \
  --alpha -20 \
  --alpha -10 \
  --alpha -5 \
  --alpha 0 \
  --alpha 5 \
  --alpha 10 \
  --alpha 20 \
  --max-alpha-trials 256 \
  --run-generation-steering \
  --max-generation-trials 96 \
  --qualitative-examples-per-alpha 3 \
  --override model=Qwen/Qwen3-8B \
  --override trust_remote_code=false
```

```bash
python scripts/run_number_game_steering_prep.py \
  --config configs/number_game/runpod_qwen3.yaml \
  --out outputs/number_game_prompt100/qwen3_8b_steering_eval_svd_subspace \
  --case-source auto \
  --max-cases 176 \
  --targets-per-clue 8 \
  --socials-per-target 2 \
  --case-seed 0 \
  --test-frac 0.25 \
  --fit-split train \
  --direction-method svd_subspace \
  --subspace-rank 3 \
  --memory-strength 1 \
  --memory-strength 4 \
  --memory-strength 8 \
  --m 1 \
  --m 3 \
  --layer 20 \
  --layer 22 \
  --layer 24 \
  --layer 26 \
  --layer 28 \
  --alpha -20 \
  --alpha -10 \
  --alpha -5 \
  --alpha 0 \
  --alpha 5 \
  --alpha 10 \
  --alpha 20 \
  --max-alpha-trials 256 \
  --run-generation-steering \
  --max-generation-trials 96 \
  --qualitative-examples-per-alpha 3 \
  --override model=Qwen/Qwen3-8B \
  --override trust_remote_code=false
```

For each run, inspect:

```bash
cat OUT_DIR/behavioral_steering_effect_summary.csv
ls OUT_DIR/plots
```
