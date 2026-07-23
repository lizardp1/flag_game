# Number Game

The number game is a text-only analogue of the flag game. Agents infer one
hidden integer. Each agent sees one private clue, such as `the number is prime`,
`the number is under 20`, or `the number is odd`, then communicates through the
same protocol families as the flag game.

Active configs live in `configs/number_game/`:

- `local_qwen3_1_7b.yaml`: small local Qwen3-1.7B smoke runs through Hugging Face Transformers.
- `runpod_qwen3.yaml`: Qwen3 RunPod sweep template; override `model`, `N`, `H`, `T`, or `protocol`.
- `runpod_kimi_k2_endpoint.yaml`: Kimi K2 through a local OpenAI-compatible vLLM/SGLang endpoint.

Do not add one config per protocol or model size. Use CLI overrides.

The rerun checklist for the prompted `1..100` experiment contract is in
`docs/number_game_rerun_checklist.md`.
RunPod setup and command details are in `docs/RUNPOD_NUMBER_GAME.md`.

All active configs use a prompted range:

```yaml
min_number: 1
max_number: 100
prompt_number_range: true
prompt_social_susceptibility: false
early_stop_window: 5
```

This means the prompt tells the model:

```text
The hidden integer is in the range 1 through 100, inclusive.
```

It does not print an allowed-number list, and parser validity does not require
answers to lie inside `1..100`.

## Protocol

- `m=1`: agents may only say a number, `{"number":17}`.
- `m=3`: agents say a number plus one sentence, `{"number":17,"reason":"It fits my prime clue and the transcript."}`.
- `pairwise`: one speaker sends one message to one listener at each step.
- `broadcast`: every agent broadcasts, then each agent makes a final decision.
- `org`: observers send clue-conditioned statements to a manager.

Pairwise memory matches the flag-game surface format. Within one run the
format is determined by `m`: `m=1` memory is number-only, and `m=3` memory is
number plus reason after a pipe.

```text
17
```

```text
17 | It fits my prime clue.
```

No speaker prefix like `A02 said ...` is included.

## Local Smoke

Run the first local check on Qwen3-1.7B:

```bash
python -m nnd.number_game.cli compare-pairwise-m \
  --config configs/number_game/local_qwen3_1_7b.yaml \
  --out results/number_game/local_qwen3_1_7b_pairwise_m_compare \
  --start-seed 0 \
  --num-seeds 5 \
  --m 1 \
  --m 3
```

Run the social/private conflict probe locally:

```bash
python -m nnd.number_game.cli conflict-battery \
  --config configs/number_game/local_qwen3_1_7b.yaml \
  --out results/number_game/local_qwen3_1_7b_conflict \
  --start-seed 0 \
  --num-seeds 1 \
  --ratio-total 8 \
  --m 1 \
  --m 3
```

## RunPod Qwen

Use the same template for Qwen3 sizes:

```bash
python -m nnd.number_game.cli conflict-battery \
  --config configs/number_game/runpod_qwen3.yaml \
  --out results/number_game/qwen3_4b_conflict \
  --start-seed 0 \
  --num-seeds 5 \
  --ratio-total 8 \
  --m 1 \
  --m 3 \
  --override model=Qwen/Qwen3-4B
```

Actual social-interaction sweep:

```bash
python -m nnd.number_game.cli compare-pairwise-m \
  --config configs/number_game/runpod_qwen3.yaml \
  --out results/number_game/qwen3_4b_pairwise_m_compare \
  --start-seed 0 \
  --num-seeds 5 \
  --m 1 \
  --m 3 \
  --override model=Qwen/Qwen3-4B \
  --override N=8 \
  --override H=8 \
  --override T=160
```

For broadcast or org parity runs, keep the same config and override protocol:

```bash
python -m nnd.number_game.cli run \
  --config configs/number_game/runpod_qwen3.yaml \
  --out results/number_game/qwen3_4b_broadcast_seed0 \
  --seed 0 \
  --protocol broadcast \
  --override model=Qwen/Qwen3-4B \
  --override interaction_m=3
```

## Kimi Endpoint

Start Kimi K2 with vLLM/SGLang or another OpenAI-compatible server, then run:

```bash
export NND_MODEL_BASE_URL=http://localhost:8000/v1
export NND_MODEL_API_KEY=local

python -m nnd.number_game.cli compare-pairwise-m \
  --config configs/number_game/runpod_kimi_k2_endpoint.yaml \
  --out results/number_game/kimi_k2_pairwise_m_compare \
  --start-seed 0 \
  --num-seeds 5 \
  --m 1 \
  --m 3
```

This uses plain HTTP; it does not require the OpenAI Python client.

## Outputs

Pairwise runs with `probe_every` write:

- `probes.csv`: all-agent belief probes over time.
- `probe_changes.csv`: explicit belief-change events.
- `probe_agent_timeline.txt`: one line per agent; `*` marks a change.
- `number_share_timeline.txt`: ASCII number-share bars.
- `plots/number_share_trajectories.svg`: line chart over probe rounds.
- `dialogues.md`: readable pairwise transcript with reasons for `m=3`.
- `social_influence_summary.json`: accuracy, consensus, change, and clue-satisfaction metrics.

Conflict probes write:

- `conflict_trials.csv`
- `conflict_summary.csv`
- `conflict_phase_summary.csv`
- `clue_information.csv`

For hidden states, use the Transformers backend and enable:

```bash
--override capture_hidden_states=true
--override hidden_state_layers=[-1]
```
