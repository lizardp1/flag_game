# Current Flag Game theory

This folder is the single current theory package. It contains the accepted
binary model, manuscript text, final four-panel figure, measurements, and
everything needed to reproduce the figure without accessing older folders or
making model API calls.

## Files to use

| File | Purpose |
| --- | --- |
| `theory_section.tex` | Current main theory text |
| `theory_appendix.tex` | Exact stationary law, fixation, and measurement methods |
| `figure_caption.tex` | Caption and figure inclusion used by the main text |
| `../paper/figures/final/fig8.pdf` | Final publication figure |
| `../paper/figures/final/fig8.png` / `.svg` | Preview and editable vector export |
| `references.bib` | References cited by the theory text |
| `preview.tex` | Standalone LaTeX reading copy |
| `make_figure.py`, `style.py` | Figure and visual style |
| `model.py` | Exact binary stationary and fixation probabilities |
| `validate.py` | Numerical and measurement checks |
| `data/` | Packaged measurements, example flag, and numerical tables |
| `validation/` | Validation results and source provenance |

## Accepted version

The figure uses the Yemen-Austria example, with pooled rival evidence share
0.655. Panel (b) is the fixed-share slice 0.35. The model has total anchor
probability 0.45 and adoption bias h0=0.3. Panel (c) is the theoretical phase
diagram. Panel (d) shows empirical soft regions with Gaussian bandwidth 0.05
in evidence share and maximum opacity 0.46, without dots or a separate legend. The latest larger fonts
and left-shifted titles are retained.

The binary theory uses three exhaustive categories: correct consensus at truth
share >=0.85, wrong consensus at truth share <=0.15, and polarization for every
remaining binary split. Both theoretical panels use the exact discrete copying
model. When both zealot types are present, endpoint probabilities sum the exact
stationary birth–death distribution over the three truth-share categories.
One/no-zealot populations use exact fixation probabilities; populations with no
ambiguous agents have fixed outcomes. Mean truth shares use the same mixture.

The seven populations N=2,4,8,16,32,64,128 are evaluated on the full
evidence-share grid (0 to 1 in steps of 0.001). Panel (b) has no point markers.
Shape-preserving cubic interpolation (PCHIP) in log2 N, normalized across
outcomes, connects the seven calculated populations without moving their values.
The connecting curves and region boundaries are display interpolation of those
exact probabilities.
Empirical panel (d) retains
its original multicountry classification, including fragmentation; its
polarization threshold remains 25% per country. The caption states this distinction.

The empirical map uses the selected 217 runs. Their outcomes are unchanged:
101 correct consensus, 15 wrong consensus, 93 polarization, and 8 fragmentation.
All 9,448 crop placements from the 227-run binary probe sweep are included as
compact count records so the selected cohort, ten exclusions, and evidence
shares can be checked locally. Full original model-call logs remain in the
repository's `results/flag_game/` directory; they are not needed for these builds.

## Reproduce

Use Python 3 with the packages in `requirements.txt`; install them with
`python3 -m pip install -r requirements.txt` if needed. Exact typography uses
Arial. Run inside this folder:

```sh
python3 make_figure.py
python3 make_social_circuit.py
python3 validate.py
```

The first command writes the three figure formats to `../paper/figures/final/`. The second
checks the exact stationary recurrence, absorbing fixation, neutral-copying
identities, the full stored probability grid, probe counts, cohort, and run-level
model references. To regenerate the model probability grid:

```sh
python3 model.py --recompute
```

The model grid contains the seven population sizes above. All probabilities use
the exact discrete calculation; PCHIP is applied only when drawing the figure.

For requested visual changes, edit `style.py` or the existing figure script and
rebuild `../paper/figures/final/fig8.*` in place. Use `python3 make_figure.py --formats png`
for a quick visual check. Keep the accepted parameters above unless the scientific
request changes them; do not create dated or version-numbered theory folders.
Temporary comparisons belong in an ignored output location outside this package.

To compile the reading copy with LaTeX and BibTeX, run `make preview`.
For manuscript integration, copy the three theory `.tex` files, the figure under
`figures/`, and merge `references.bib`. Preserve these relative paths, or adjust
the inputs to match the manuscript's directory structure.


## A4 mean earliest arrival

Run `python make_social_circuit.py` to rebuild Figure 6 from the bundled schedule.
The mean over seven peers, excluding A4, is 90/(7×8) = 1.607142857 rounds.
This measures potential temporal reach, not observed belief adoption.
