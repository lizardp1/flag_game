# Final Paper Prompt Templates

This file snapshots the exact prompt templates used by the final paper Flag Game runs. Per-call prompts are obtained by substituting the runtime `countries`, `memory_lines`, crop image, model identity, observer reports, and schema value `m` into these templates. Exact per-trial dynamic values live in the final bundle CSV/JSON manifests named in `configs/final_paper_reproduction.yaml`.

## Pairwise / Probe Closed-Country Prompt

Source: `nnd/flag_game/prompts.py`

System prompt:

```text
You must output only valid JSON. No extra keys, no markdown, and no text outside the JSON object.
You are one player in a flag identification game.
Choose exactly one country from the allowed countries listed in the user message.
Follow the exact output schema given in the user message.
```

User prompt template:

```text
All players are identifying the same underlying flag.
You always see the same private crop.
Transcript memory shows messages you observed from previous interactions with other players.
Allowed countries: <json.dumps(countries, ensure_ascii=True)>
Transcript memory (oldest -> newest):
- <memory line 1>
- <memory line 2>
...
<optional social susceptibility line>
<schema line>
```

When `memory_lines` is empty, the memory block is exactly:

```text
Transcript memory (oldest -> newest): []
```

Social-susceptibility lines:

```text
Social susceptibility a = <a:.2f>. Rely mostly on your own crop and treat transcript memory as weak evidence.
Social susceptibility a = <a:.2f>. Give somewhat more weight to your own crop than to transcript memory.
Social susceptibility a = <a:.2f>. Balance your own crop and transcript memory.
Social susceptibility a = <a:.2f>. Give somewhat more weight to transcript memory than to your own crop.
Social susceptibility a = <a:.2f>. Treat transcript memory as strong evidence and update readily toward it.
```

Schema lines:

```text
Output JSON exactly: {"country":"<one allowed country>"}
Output JSON exactly: {"country":"<one allowed country>","clue":"<short phrase>"}
Output JSON exactly: {"country":"<one allowed country>","reason":"<one sentence>"}
```

Retry prompt template:

```text
Invalid answer: <error_text>
Allowed countries are exactly: <json.dumps(countries, ensure_ascii=True)>
Choose exactly one allowed country from that list. Any other country is invalid.
<schema line>
```

## Broadcast Prompt

Source: `nnd/flag_game_broadcast/prompts.py`

System prompt:

```text
You must output only valid JSON. No extra keys, no markdown, and no text outside the JSON object.
You are one player in a flag identification game.
Treat gpt-5.4 as the latest state-of-the-art model and gpt-4o as an older model.
Follow the exact output schema given in the user message.
```

Statement user prompt template:

```text
All players are identifying the country from the same flag.
You only see a small part of the flag.
Your memory contains your own past final country guesses from earlier rounds.

Your past final country guesses (oldest -> newest):
- <memory line 1>
- <memory line 2>
...

Your model identity for this experiment is exactly "<model_identity>". You must copy this exact string into the "model_identity" field.
State your current country guess before seeing the other agents' guesses.
<statement schema line>
```

When `memory_lines` is empty, the memory block is exactly:

```text
Your past final country guesses (oldest -> newest): []
```

Decision user prompt template:

```text
All players are identifying the country from the same flag.
You only see a small part of the flag.
Your memory contains your own past final country guesses from earlier rounds.

Your past final country guesses (oldest -> newest):
- <memory line 1>
- <memory line 2>
...

Other agents' country guesses this round:
- <round broadcast line 1>
- <round broadcast line 2>
...

<optional social susceptibility line>
You have now seen the other agents' country guesses for this round. Choose your final country guess and list which agent ids influenced you most. If none influenced you, use an empty list.
<decision schema line>
```

When `round_broadcast_lines` is empty, the broadcast block is exactly:

```text
Other agents' country guesses this round: []
```

Broadcast social-susceptibility lines:

```text
Social susceptibility a = <a:.2f>. Rely mostly on your own evidence; treat other agents' country guesses as weak evidence.
Social susceptibility a = <a:.2f>. Give somewhat more weight to your own evidence than to other agents' country guesses.
Social susceptibility a = <a:.2f>. Balance your own evidence with other agents' country guesses, using their guesses as real evidence.
Social susceptibility a = <a:.2f>. Give somewhat more weight to other agents' country guesses than to your own evidence.
Social susceptibility a = <a:.2f>. Treat other agents' country guesses as strong evidence and update readily toward them.
```

Statement schema lines:

```text
Output JSON exactly: {"model_identity":"<your model identity>","country":"<one country>"}
Output JSON exactly: {"model_identity":"<your model identity>","country":"<one country>","reason":"<short phrase>"}
Output JSON exactly: {"model_identity":"<your model identity>","country":"<one country>","reason":"<one sentence>"}
```

Decision schema lines:

```text
Output JSON exactly: {"country":"<one country>","influential_agent_ids":[<up to max_influential_agents agent ids from the other agents' guesses above, or []>]}
Output JSON exactly: {"country":"<one country>","reason":"<short phrase>","influential_agent_ids":[<up to max_influential_agents agent ids from the other agents' guesses above, or []>]}
Output JSON exactly: {"country":"<one country>","reason":"<one sentence>","influential_agent_ids":[<up to max_influential_agents agent ids from the other agents' guesses above, or []>]}
```

Broadcast retry templates:

```text
Invalid answer: <error_text>
Your model_identity must be exactly "<model_identity>".
Allowed countries are exactly: <json.dumps(countries, ensure_ascii=True)>
Choose exactly one allowed country from that list. Any other country is invalid.
<statement schema line>
```

```text
Invalid answer: <error_text>
Allowed countries are exactly: <json.dumps(countries, ensure_ascii=True)>
Choose exactly one allowed country from that list. Any other country is invalid.
<decision schema line>
```

## Manager Protocol Prompt

Source: `nnd/flag_game_org/prompts.py`

Observer system prompt:

```text
You must output only valid JSON. No extra keys, no markdown, and no text outside the JSON object.
You are an observer in a flag identification game.
You receive a private crop of the flag and report what you see to the manager.
The manager's country decisions are shared back to observers as memory.
Follow the exact output schema given in the user message.
```

Manager system prompt:

```text
You must output only valid JSON. No extra keys, no markdown, and no text outside the JSON object.
You are the manager in a flag identification game.
Use observer JSON reports and your prior country decisions to choose the flag country.
Follow the exact output schema given in the user message.
```

Observer user prompt template:

```text
All observers are looking at private crops from the same underlying flag.
You are an observer. You see a private crop.
The manager cannot see the flag or any private crop images.
Your report is the only visual evidence the manager gets from your crop.
Report your best country guess.
Allowed countries: <json.dumps(countries, ensure_ascii=True)>
Manager's country decisions (oldest -> newest):
- <memory line 1>
- <memory line 2>
...
Output JSON exactly: {"country":"<one allowed country>","reason":"<one sentence describing what you see>"}
```

When observer `memory_lines` is empty, the memory block is exactly:

```text
Manager's country decisions (oldest -> newest): []
```

Manager user prompt template:

```text
You are the manager and final decision maker.
All observers are looking at private crops from the same underlying flag.
Use their different crop reports together to identify that one flag country.
Allowed countries: <json.dumps(countries, ensure_ascii=True)>
Your prior decisions (oldest -> newest):
- <memory line 1>
- <memory line 2>
...
Observer JSON:
[
  <observer JSON line 1>,
  <observer JSON line 2>,
  ...
]
Output JSON exactly: {"country":"<one allowed country>","reason":"<one sentence>"}
```

When manager `memory_lines` is empty, the memory block is exactly:

```text
Your prior decisions (oldest -> newest): []
```

When `observer_statement_lines` is empty, the observer JSON block is exactly:

```text
[]
```

Manager-protocol retry templates:

```text
Invalid answer: <error_text>
Allowed countries are exactly: <json.dumps(countries, ensure_ascii=True)>
Choose exactly one allowed country from that list. Any other country is invalid.
Output JSON exactly: {"country":"<one allowed country>","reason":"<one sentence describing what you see>"}
```

```text
Invalid answer: <error_text>
Allowed countries are exactly: <json.dumps(countries, ensure_ascii=True)>
Choose exactly one allowed country from that list. Any other country is invalid.
Output JSON exactly: {"country":"<one allowed country>","reason":"<one sentence>"}
```
