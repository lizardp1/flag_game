# Protocols and controls

The single validated configuration schema is `nnd.experiment_cli.Experiment`. Unknown keys and unsupported provider/protocol combinations fail before calls.

| Control | Interpretation |
| --- | --- |
| `N` | Crop-bearing observers. Manager adds one blind agent: total N+1. |
| `composition` | Model-to-positive-integer counts summing to N. No rounding of proportions. |
| `manager_model` | Separate manager model; ignored outside manager protocol. |
| `social_evidence_alpha` | Selects one of five verbal guidance clauses, bounded [0,1]. Not a calibrated behavioral weight. |
| `social_guidance_enabled` | False omits guidance entirely; alpha=.5 with guidance is a different condition. |
| `message_bandwidth` | 1 = country only; 2 = short clue/phrase; 3 = one-sentence reason. |
| `memory_capacity` | FIFO history capacity H; zero disables retained entries. |
| `rounds` | Pairwise T=rounds*N interactions, readouts every N; broadcast/manager synchronous rounds. |
| `seeds` | Exact crop/target/schedule random seeds. API sampling remains stochastic. |
| `randomize_model_slots` | Seeded anonymous model assignment, recorded only in internal metadata. |
| `early_stop_window` | 0 disables stopping. Pairwise/broadcast require unanimous valid reports over the window; manager stops after repeated manager decisions. These are different stopping semantics. |

## Pairwise

A speaker uses its crop and received-message memory. Its valid message is appended to the listener's FIFO. Initial guesses and later probes measure population performance. m=2 uses the legacy `clue` JSON key; the other protocols use `reason` for their short phrase. Pairwise probes after initialization remain country-only readouts and do not feed back into memory.

## Anonymous broadcast

Every agent gives a report, reads valid peer reports, and gives its final decision. Closed allowed-country lists occur in both stages. Agent-visible content contains no model identity, SOTA framing, agent IDs, or numeric alpha label. Guidance uses the exact pairwise verbal clauses. Reports and private memory use `country | reason` for reason-bearing bandwidths and country only for m=1. Each agent retains only its own final decisions. Invalid peer reports are omitted. Model assignments remain in internal manifests.

## Manager extension

N observers have private crops; one additional manager never receives images. Each round the manager sees observer reports and its own prior decisions. Its valid decision is broadcast into observers' shared FIFO memory. The final metric is manager correctness, not population consensus.

All three bandwidths apply to observer reports **and manager feedback**. m=1 carries no hidden reason field. m=2 carries a short phrase; m=3 carries a sentence. Strict parsers reject wrong keys, but sentence/phrase length is a prompt constraint, as in the existing protocols.

With guidance enabled, observer alpha balances **private crop versus manager feedback**. Manager alpha balances **prior decisions versus current observer reports**. High alpha therefore increases each role's uptake of its current social source; it does not pretend the blind manager has visual evidence. At the first round the manager has no previous decisions, so alpha is not a promise of a first-round behavioral difference.

Manager configurations default to guidance disabled and m=3. Enable social guidance to apply alpha.

## Outcomes

Public population presets use consensus >=.85 and polarization requiring at least two countries each >=.25 below consensus. Manager endpoint accuracy is its final answer; observer readouts are separate diagnostics. The binary theory has three exhaustive categories; empirical multicountry outcomes retain fragmentation. Do not apply the theory's residual-split classification to empirical runs.

OpenAI supports all three protocols. Anthropic currently supports pairwise and probes. Cross-provider teams are rejected by the common interface. Scripted runs test plumbing only; their heuristic social weighting is not a model of real alpha response.
