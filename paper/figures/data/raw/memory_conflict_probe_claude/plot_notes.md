# Memory Conflict Probe Plot Notes

## Choice Axis Definition

- `0 = target country`: the agent chose the hidden target/original country that generated the private crop.
- `1 = social evidence`: the agent chose the synthetic lure country supported by transcript memory.
- In strong private evidence, the target country is also the only country compatible with the crop.
- In weak private evidence, the target country is only one of several countries compatible with the crop. So `0` means target recovery, not unique visual certainty or the only private-image-supported answer.
- The choice-axis plots use only trials where the agent chose either the target country or the social-evidence country. Other crop-compatible answers are not placed on this binary axis.

## Memory Composition Axis

- X-axis tick labels show `target memories : social-evidence memories`.
- For example, `6:2` means six synthetic memory entries support the target country and two support the social-evidence country.
- Moving right means less target-country memory support and more social-evidence memory support.

## Private Evidence Strength

- `Strong private evidence`: the private crop can be produced by exactly one allowed country. In this run, that country is the truth/private-evidence country.
- `Weak private evidence`: the private crop can be produced by many allowed countries, including the truth/private-evidence country.
- Strength is computed mechanically by rendering every allowed flag, enumerating all same-sized crop positions, and checking which countries can produce the exact private-crop pixels.

## m Definition

- `m=1`: label-only memory entries and label-only model outputs.
- `m=3`: memory entries include a country plus a one-sentence reason, and outputs also include a reason.

## Response Decomposition

- `Private target country`: the agent chose the hidden target/original country that generated the private crop.
- `Social evidence`: the agent chose the synthetic social-evidence lure.
- `Other crop-compatible country`: the agent chose a different country that can produce the same private-crop pixels. This is still consistent with the private image, but it is not the hidden target/original country.
- `Unsupported other`: the agent chose a third country that is neither the social lure nor compatible with the private crop.
- If `--lure-relation compatible` is included, the social-evidence country also fits the private crop. This is possible for weak private evidence, but not for strong private evidence where only the target country fits.

## Evidence Alignment Decomposition

- `Private target country`: the agent chose the hidden target/original country that generated the private crop.
- `Social evidence`: the agent chose the synthetic social-evidence lure.
- `Other: private + social compatible`: a third-country answer compatible with the private image and also compatible with the social-evidence country's flag features.
- `Other: private compatible`: a third-country answer compatible with the private image, but not compatible with the social-evidence country's flag features.
- `Other: social compatible`: a third-country answer not compatible with the private image, but compatible with the social-evidence country's flag features.
- `Other: incompatible`: a third-country answer compatible with neither the private image nor the social-evidence country's flag features.
- Social compatibility is a coarse heuristic: same stripe orientation with at least two shared colors, or matching triangle features. Exact row-level cases are written to `off_axis_choice_audit.csv`.

## Memory Order

- Each trial has 8 synthetic memory entries: `false_memory_count` entries for the social-evidence country and `8 - false_memory_count` entries for the target country.
- The memory-entry order is shuffled independently for each replicate. The allowed-country list is also shuffled independently for each trial.

## Current Run Note

- Overall third-country answer rate: `0.053`. These answers are still in `results.csv`, but excluded from binary private-vs-social choice-axis means.

Recommended visual for interpretation: `agent_response_decomposition.png`, plus the `m1` and `m3` variants.

`private_vs_social_evidence_choice_axis.png` is a narrower binary view. It excludes third-country answers and asks only: among runs where the agent chose either the target country or the social-evidence country, how often did it choose the social-evidence country?