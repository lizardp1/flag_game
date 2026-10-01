# Supporting methods and scope

## Intervention and outcome

The Germany study patches **private crop images** before initialization and at subsequent model calls. The donor is source A1's already existing, calibrated view of the same flag; source A4's original crop shows black and red, while the donor shows red and a narrow yellow region. Native crop dimensions are 150 × 100 RGB pixels. The patch does not change the flag's truth label or directly edit memory, model activations, or reported beliefs. Memory and messages subsequently evolve under the ordinary protocol.

The requested model was `gpt-4o`; the recorded provider model was `gpt-4o-2024-08-06`. Social games use country-and-reason messages (m=3), memory length H=8, a closed 28-country stripe/triangle list, temperature 0.2, top_p 1, and high image detail. Country-only probes measure collective mean accuracy. The primary population comparison is **original versus patched at R10**, not pre-communication versus post-communication within a condition. At R10 there have been 10N directed messages. The data also retain R20 readouts.

Original and patched conditions share their directed communication schedule within each population-size/seed comparison. Responses are fresh provider draws: matching schedules does not imply identical API randomness. The intervention effect includes changes to the patched agents and any subsequent social propagation; it does not isolate a mediated-only effect.

## Site selection and traces

The saved eight-crop layout was chosen from earlier wrong-answer examples. In the site screen, each of A0–A7 receives the same donor in a separate trial with schedule seed 73 (repeat 050). Their R10 collective mean accuracies are 75%, 25%, 37.5%, 37.5%, 100%, 25%, 37.5%, and 87.5%. A1 is an identity control. Three original controls (repeats 050–052) each have 25% collective mean accuracy.

A4 was selected by the largest gain in correct answers among unchanged agents, then total gain, then lowest agent ID among non-identity sites. A4 also has the largest whole-population collective mean accuracy, as shown in the figure. The manuscript therefore describes the observed selection using the displayed outcome, without claiming that this was a preregistered ranking criterion. One run per site does not distinguish a persistent agent-site effect from schedule and response variability.

The displayed pair uses schedule seed 73, repeat 600. It is a separate pair from the site screen, but is included in the N=8 repeated-run aggregate (repeats 600–609). At the displayed R10 endpoint, the original arm has two Germany and six Yemen answers, and the patched arm has eight Germany answers. The original arm is not unanimous wrong consensus. The pair and displayed endpoint were selected retrospectively from available trajectories.

For an arrow into recipient j at displayed round r, j must change country between probes r−1 and r. The latest valid incoming message received by j during that interval must name j's new country, and its sender's probe at r−1 must name that country too. There is no fallback to an older message when the latest message fails these conditions. This rule provides message-compatible transmission candidates. The study does not independently ablate or patch every displayed message edge.

## Population comparisons and uncertainty

- **N=1:** ten independent empty-memory, country-only judgments per crop, with no communication, interaction schedule, or rounds. The original crop gives Yemen ten times; the informative crop gives Germany ten times. One patch changes 100% of the population. The observed bootstrap intervals are degenerate because every observed answer agrees; this does not establish zero uncertainty for future judgments.
- **N=4:** the previously selected subset 4 contains source A0, A1, A2, and A4. Local A3 (source A4) is patched, changing 25% of the population. The subset was chosen after reviewing six configurations.
- **N=8–128:** repeat the same eight source crops and patch source-A4 copies at local IDs 4, 12, 20, …, N−4. This keeps the empirical crop proportions fixed within each condition and changes 12.5% of the population.

There are ten social runs per condition at each N≥4, using schedule seeds 73–82. Error bars in the figure are 95% percentile bootstrap intervals for each condition mean, resampling whole runs (judgments at N=1), using 200,000 resamples and RNG seed 20260908. Intervals for the effects quoted in the text resample differences between schedule-matched runs. The N=1 judgments have no paired-schedule effect interval. These intervals do not account for layout, site, or display-window selection, and are not estimates of the model's N^(−1/2) dynamical noise scaling.

| Population size | Original collective mean accuracy | Patched collective mean accuracy | Patching effect |
|---|---:|---:|---:|
| 1 | 0% | 100% | +100 points |
| 4 | 40% | 82.5% | +42.5 points |
| 8 | 30% | 70% | +40 points |
| 16 | 33.125% | 61.25% | +28.125 points |
| 32 | 29.6875% | 50.3125% | +20.625 points |
| 64 | 29.53125% | 46.875% | +17.34375 points |
| 128 | 28.203125% | 45.3125% | +17.109375 points |

## Relationship to the theory

The finite-population model in the theory section draws evidence types independently with probabilities a_T, a_R, and a_0. Its evidence-appearance probabilities concern the chance of obtaining previously absent types. The repeated-layout experiment does not change evidence diversity this way at N≥8. It shows an empirical change in the finite-horizon response to an intervention while crop proportions and intervention fraction are fixed.

The intervention curve therefore does not quantitatively validate the evidence-coverage mechanism, demonstrate loss of causal identifiability, or establish a threshold beyond which mechanistic interpretability fails. It motivates a complementary population-level analysis of a measured intervention response. Positive patching effects remain at large N. Explaining their magnitude and size dependence would require fitting/testing finite-population dynamics or a richer agent update model.
