# Final Paper Chart Bundle

This folder contains the figures actually used in the paper, their plotting scripts, the relevant run scripts used to obtain the data, and the underlying data needed by those plotting scripts.

The bundle mirrors the repository layout below this directory:

- `paper/exports/figures/`: final chart files copied by stem.
- `paper/exports/data/`: compact source and exported summary tables/JSON used by final chart scripts.
- `paper/*.py`: plotting scripts for the final charts.
- `scripts/`: data-generation or convenience scripts associated with the final runs.
- `results/`: slimmed source data required by the final plots. Full per-seed image artifact dumps are intentionally omitted; only the small set of stimulus PNGs needed to redraw image-bearing panels is retained.
- `poster/exports/support_visuals/`: broadcast summary source used by the broadcast and slot figures.

## Final Chart Stems

- `agent_memory_probe_2`
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

## Rebuilding

From this folder, run:

```bash
./scripts/rebuild_final_charts.sh
```

The protocol and slot protocol path uses the compact final ranked-performance table at `paper/exports/data/flag_gam_protocol_side_by_side_N8_final_ranked_performance.csv`; the full raw protocol seed folders are intentionally not shipped in this bundle.

The broadcast slot mini-panels use `paper/exports/data/flag_game_slot_broadcast_compact_summary.json` for the compact mean/SEM values needed to redraw the shaded bands without restoring the raw sweep tree.

`agent_memory_probe_2.pdf` is included as the archived final paper asset. A source generator for that exact stem was not identifiable in the repository search; the related paper figure reference points at `agent_memory_probe.pdf`.

See `manifest.json` for the chart-to-script/data mapping.

## Exact Run Metadata

Repo-level reproducibility metadata is in:

- `../../configs/final_paper_reproduction.yaml`: exact model IDs, API settings, trial counts, seed inventories, data sources, and figure commands.
- `../../configs/final_paper_prompt_templates.md`: exact prompt templates for pairwise/probe, broadcast, and manager-protocol runs.
