#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

"$PYTHON" paper/make_flag_protocol_side_by_side_visuals.py \
  --source-ranked-csv paper/exports/data/flag_gam_protocol_side_by_side_N8_final_ranked_performance.csv \
  --out-stem flag_gam_protocol_side_by_side_N8_final \
  --final-only \
  --x-max 0.8

"$PYTHON" paper/make_flag_game_slot_figures.py

"$PYTHON" paper/make_flag_broadcast_visuals.py

"$PYTHON" paper/make_flag_visual_only_audit_visuals.py \
  --source results/flag_game/visual_only_paired_crop_audit/results.csv \
  --figure-dir paper/exports/figures \
  --data-dir paper/exports/data \
  --stem flag_game_visual_only_paired_audit

FINAL_CHARTS_DERIVED_ONLY=1 "$PYTHON" paper/make_flag_game_alpha_visuals.py

FINAL_CHARTS_DERIVED_ONLY=1 "$PYTHON" paper/make_flag_game_n_scaling_visuals.py

"$PYTHON" paper/make_flag_memory_conflict_probe_visuals.py \
  --source-csv results/flag_game/memory_conflict_probe_pilot/results.csv \
  --source-csv results/flag_game/memory_conflict_probe_claude/results.csv \
  --source-csv results/flag_game/memory_conflict_probe_opus/results.csv \
  --source-csv results/flag_game/memory_conflict_probe_opus46/results.csv \
  --stem-prefix flag_game_memory_conflict_probe_all_models \
  --alignment-only

"$PYTHON" paper/make_flag_memory_conflict_probe_visuals.py \
  --source-csv results/flag_game/memory_conflict_probe_pilot/results.csv \
  --source-csv results/flag_game/memory_conflict_probe_claude/results.csv \
  --stem-prefix flag_game_memory_conflict_probe_all_models_m1_contradicts_no_opus \
  --m 1 \
  --facet weak-incompatible \
  --alignment-only \
  --hide-section-titles

for ext in pdf png svg; do
  cp \
    "paper/exports/figures/flag_game_memory_conflict_probe_all_models_m1_contradicts_no_opus_alignment.${ext}" \
    "paper/exports/figures/flag_game_memory_conflict_probe_all_models_alignment_m1_contradicts_no_opus.${ext}"
done

./scripts/make_flag_mechanism1_measured_field_final.sh

for path in paper/exports/figures/*; do
  [[ -f "$path" ]] || continue
  name="$(basename "$path")"
  case "$name" in
    agent_memory_probe_2.pdf|\
    flag_game_memory_conflict_probe_all_models_alignment_m1_contradicts_no_opus.pdf|\
    flag_game_memory_conflict_probe_all_models_alignment_m1_contradicts_no_opus.png|\
    flag_game_memory_conflict_probe_all_models_alignment_m1_contradicts_no_opus.svg|\
    flag_gam_protocol_side_by_side_N8_final.pdf|\
    flag_gam_protocol_side_by_side_N8_final.png|\
    flag_game_memory_conflict_probe_all_models_alignment.pdf|\
    flag_game_memory_conflict_probe_all_models_alignment.png|\
    flag_game_memory_conflict_probe_all_models_alignment.svg|\
    flag_game_mechanism1_measured_field_final_france_peru_gpt4o_B50_seed0003.pdf|\
    flag_game_mechanism1_measured_field_final_france_peru_gpt4o_B50_seed0003.png|\
    flag_game_mechanism1_measured_field_final_france_peru_gpt4o_B50_seed0003.svg|\
    flag_game_broadcast_m3_diversity_main_with_empirical_metrics.pdf|\
    flag_game_broadcast_m3_diversity_main_with_empirical_metrics.png|\
    flag_game_visual_only_paired_audit_main_half_height.pdf|\
    flag_game_visual_only_paired_audit_main_half_height.png|\
    flag_game_visual_only_paired_audit_main_half_height.svg|\
    flag_game_pairwise_n_scaling_france_peru_mechanism_N4_N16_N64_with_empirical_ab_failure_decomposition.pdf|\
    flag_game_pairwise_n_scaling_france_peru_mechanism_N4_N16_N64_with_empirical_ab_failure_decomposition.png|\
    flag_game_pairwise_n_scaling_france_peru_mechanism_N4_N16_N64_with_empirical_ab_failure_decomposition.svg|\
    flag_game_slot_protocol_side_by_side_N8_mixed_ciq.pdf|\
    flag_game_slot_protocol_side_by_side_N8_mixed_ciq.png|\
    flag_game_slot_protocol_side_by_side_N8_mixed_ciq.svg|\
    flag_game_slot_broadcast_m3_diversity_mean_ciq.pdf|\
    flag_game_slot_broadcast_m3_diversity_mean_ciq.png|\
    flag_game_slot_broadcast_m3_diversity_mean_ciq.svg|\
    flag_game_slot_broadcast_m3_alpha_mean_ciq.pdf|\
    flag_game_slot_broadcast_m3_alpha_mean_ciq.png|\
    flag_game_slot_broadcast_m3_alpha_mean_ciq.svg|\
    flag_game_slot_pairwise_n_scaling_mean_ciq.pdf|\
    flag_game_slot_pairwise_n_scaling_mean_ciq.png|\
    flag_game_slot_pairwise_n_scaling_mean_ciq.svg|\
    flag_game_pairwise_alpha_main_candidate.pdf|\
    flag_game_pairwise_alpha_main_candidate.png)
      ;;
    *)
      rm "$path"
      ;;
  esac
done

for path in paper/exports/data/*; do
  [[ -f "$path" ]] || continue
  name="$(basename "$path")"
  case "$name" in
    flag_gam_protocol_side_by_side_N8_final_ranked_performance.csv|\
    flag_gam_protocol_side_by_side_N8_final_summary.json|\
    flag_game_broadcast_m3_alpha_main_summary.json|\
    flag_game_broadcast_m3_diversity_main_summary.json|\
    flag_game_broadcast_m3_diversity_main_with_empirical_metrics_summary.json|\
    flag_game_slot_broadcast_compact_summary.json|\
    flag_game_mechanism1_measured_field_final_france_peru_gpt4o_B50_seed0003_abstract_field_mixture.csv|\
    flag_game_mechanism1_measured_field_final_france_peru_gpt4o_B50_seed0003_mean_field_phase_diagram.csv|\
    flag_game_mechanism1_measured_field_final_france_peru_gpt4o_B50_seed0003_finite_agent_votes.csv|\
    flag_game_mechanism1_measured_field_final_france_peru_gpt4o_B50_seed0003_finite_agent_endpoint_regimes.csv|\
    flag_game_mechanism1_measured_field_final_france_peru_gpt4o_B50_seed0003_summary.json|\
    flag_game_memory_conflict_probe_alignment_summary.csv|\
    flag_game_memory_conflict_probe_findings.md|\
    flag_game_memory_conflict_probe_summary.json|\
    flag_game_pairwise_alpha_main_candidate_summary.csv|\
    flag_game_pairwise_alpha_main_candidate_summary.json|\
    flag_game_pairwise_n_scaling_france_peru_mechanism_N4_N16_N64_with_empirical_ab_failure_decomposition_summary.json|\
    flag_game_pairwise_n_scaling_main_v3_summary.csv|\
    flag_game_pairwise_n_scaling_main_v3_summary.json|\
    flag_game_visual_only_paired_audit_crop_incompatible_rows.csv|\
    flag_game_visual_only_paired_audit_example_pairs.csv|\
    flag_game_visual_only_paired_audit_model_visual_read_categories.csv|\
    flag_game_visual_only_paired_audit_summary.json)
      ;;
    *)
      rm "$path"
      ;;
  esac
done

"$PYTHON" - "$ROOT" <<'PY'
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1]).resolve()
repo_root = root.parent.parent.resolve()
prefixes = [f"{root}/", f"{repo_root}/"]
skip_suffixes = {".png", ".pdf", ".svg", ".pyc"}

for path in root.rglob(".DS_Store"):
    path.unlink(missing_ok=True)
for path in root.rglob("*.pyc"):
    path.unlink(missing_ok=True)
for path in sorted(root.rglob("__pycache__"), key=lambda item: len(item.parts), reverse=True):
    if path.is_dir():
        shutil.rmtree(path)

for path in root.rglob("*"):
    if not path.is_file() or path.suffix.lower() in skip_suffixes:
        continue
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        continue
    updated = text
    for prefix in prefixes:
        updated = updated.replace(prefix, "")
    if updated != text:
        path.write_text(updated)
PY

echo "Rebuilt final chart bundle in $ROOT"
