# Flag Game

This directory is the compact reproduction bundle for the figures used in the paper. It contains the final figure exports, plotting scripts, relevant run scripts, compact plotted data, and the slimmed source data needed to redraw the panels.

## Layout

- `paper/exports/figures/`: final figure files used in the paper.
- `paper/exports/data/`: compact CSV/JSON source tables and exported summaries used by the plotting scripts.
- `paper/*.py`: plotting scripts.
- `scripts/`: run helpers and the one-command rebuild script.
- `results/`: slimmed source data required by the final plots. Full per-seed image artifact dumps are intentionally omitted; only the stimulus PNGs needed by image-bearing final panels are retained.
- `poster/exports/support_visuals/`: compact broadcast sweep summary used by the broadcast and slot figures.

## Rebuild

From this folder, run:

```bash
./scripts/rebuild_final_charts.sh
```

The rebuild script regenerates the final figures from the bundled compact data and slimmed source inputs, then removes helper and intermediate outputs so the folder stays paper-facing.

## Final Figure Stems

- `flag_game_memory_conflict_probe_all_models_alignment_m1_contradicts_no_opus`
- `flag_gam_protocol_side_by_side_N8_final`
- `flag_game_memory_conflict_probe_all_models_alignment`
- `flag_game_mechanism1_measured_field_final_france_peru_gpt4o_B50_seed0003`
- `flag_game_broadcast_m3_diversity_main_with_empirical_metrics`
- `flag_game_visual_only_paired_audit_main_half_height`
- `flag_game_pairwise_n_scaling_france_peru_mechanism_N4_N16_N64_with_empirical_ab_failure_decomposition`
- `flag_game_slot_protocol_side_by_side_N8_mixed_ciq`
- `flag_game_slot_broadcast_m3_diversity_mean_ciq`
- `flag_game_slot_broadcast_m3_alpha_mean_ciq`
- `flag_game_slot_pairwise_n_scaling_mean_ciq`
- `flag_game_pairwise_alpha_main_candidate`

## Compact Data Notes

The protocol side-by-side figures use `paper/exports/data/flag_gam_protocol_side_by_side_N8_final_ranked_performance.csv`, which is the final plotted 12-row table for the N=8 pairwise, broadcast, and manager comparison. The full raw protocol seed folders are intentionally not shipped in this bundle.

The broadcast slot mini-panels use `paper/exports/data/flag_game_slot_broadcast_compact_summary.json` for the compact mean/SEM values needed to redraw their shaded bands without restoring the raw sweep tree.

The pairwise alpha and N-scaling rebuild paths run in derived-only mode from compact exported summaries, plus the retained representative seed data needed by the mechanism panel.

## Exact Metadata

Repo-level reproduction metadata is in:

- `../../configs/final_paper_reproduction.yaml`: exact model IDs, API settings, trial counts, seed inventories, source data, and figure commands.
- `../../configs/final_paper_prompt_templates.md`: exact prompt templates for pairwise/probe, broadcast, and manager-protocol runs.
- `manifest.json`: chart-to-script/data mapping for the final figures.
