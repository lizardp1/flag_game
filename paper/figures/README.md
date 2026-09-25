# Paper figures

All eight complete main-paper figures are in [`final/`](final/README.md), named
`fig1` through `fig8` using the manuscript numbering.

- `final/`: complete numbered figures for reading and paper inclusion.
- `generated/`: individual data-driven panels; descriptive filenames identify their contents.
- `code/`, `data/`, `assets/`: plotting code, compact inputs, and illustration assets.
- `manifest.json`: maps final figures and panels to inputs and builders. Preserved exports have integrity hashes.

Run `bash paper/figures/build.sh` from the repository root. It rebuilds the ten
empirical components and complete Figures 6 and 8 without model calls. Figures
with manual layouts remain preserved exports; rebuilding panels does not
reassemble those layouts. 

Theory source code stays in `../../theory/`; its builders write Figures 6 and 8
directly into `final/`. Do not create dated or versioned figure folders for tweaks.
