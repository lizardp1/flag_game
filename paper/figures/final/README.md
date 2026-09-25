# Complete paper figures

Use these files for the eight main-paper figures. Numbering follows `Flag_Game_final.pdf`.

| Figure | File | Content |
|---|---|---|
| 1 | [fig1.png](fig1.png) | Flag Game overview and main results |
| 2 | [fig2.png](fig2.png) | Protocol comparison |
| 3 | [fig3.pdf](fig3.pdf) | Population scaling and polarization |
| 4 | [fig4.png](fig4.png) | Diversity and single-agent vision test |
| 5 | [fig5.pdf](fig5.pdf) | Memory probe |
| 6 | [fig6.pdf](fig6.pdf) | A4 temporal reach and mean earliest arrival |
| 7 | [fig7.pdf](fig7.pdf) | Crop-patching intervention |
| 8 | [fig8.pdf](fig8.pdf) | Binary theory |

Figures 6 and 8 also have PNG previews and editable SVGs. Their builders write
here directly. Figures 1–5 and 7 preserve complete manuscript artwork; their
available chart components rebuild in `../generated/`, without reassembling
manual layouts. Figures 3, 5, and 7 are vector forms extracted from the supplied
paper. Figure 6 includes the subsequently requested mean-arrival annotation.

Run `bash paper/figures/build.sh` from the repository root. It rebuilds the
empirical components and complete Figures 6 and 8, then checks the manifest.
Visual tweaks replace these outputs in place.
