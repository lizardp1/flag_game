#!/usr/bin/env bash
set -euo pipefail

FIGURE_ROOT="$(cd "$(dirname "$0")" && pwd)"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/flag-game-mplconfig"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/flag-game-cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$FIGURE_ROOT/generated" "$FIGURE_ROOT/data"

python3 "$FIGURE_ROOT/code/make_flag_game_slot_figures.py"

python3 "$FIGURE_ROOT/code/make_flag_broadcast_visuals.py"

python3 "$FIGURE_ROOT/code/make_flag_protocol_side_by_side_visuals.py" \
  --source-ranked-csv "$FIGURE_ROOT/data/flag_gam_protocol_side_by_side_N8_final_ranked_performance.csv" \
  --out-stem flag_gam_protocol_side_by_side_N8_final \
  --final-only

python3 "$FIGURE_ROOT/code/make_flag_visual_only_audit_visuals.py" \
  --source "$FIGURE_ROOT/data/raw/visual_only_paired_crop_audit/results_with_followups.csv" \
  --category-source "$FIGURE_ROOT/data/raw/visual_only_paired_crop_audit/results.csv" \
  --figure-dir "$FIGURE_ROOT/generated" \
  --data-dir "$FIGURE_ROOT/data" \
  --manual-adjudication "$FIGURE_ROOT/data/visual_audit_manual_adjudication.csv" \
  --selected-example-pair-id gabon__idx084__t04_l08_h04_w06 \
  --selected-example-pair-id guinea__idx232__t12_l04_h04_w06 \
  --selected-example-pair-id yemen__idx163__t08_l11_h04_w06

python3 "$FIGURE_ROOT/code/make_flag_memory_conflict_probe_visuals.py" \
  --source-csv "$FIGURE_ROOT/data/raw/memory_conflict_probe_pilot/results.csv" \
  --source-csv "$FIGURE_ROOT/data/raw/memory_conflict_probe_claude/results.csv" \
  --figure-dir "$FIGURE_ROOT/generated" \
  --data-dir "$FIGURE_ROOT/data" \
  --stem-prefix flag_game_memory_conflict_probe_all_models_m1_contradicts_no_opus \
  --model gpt-4o \
  --model gpt-5.4 \
  --model claude-haiku-4-5-20251001 \
  --model claude-sonnet-4-6 \
  --m 1 \
  --facet weak-incompatible \
  --facet strong-incompatible \
  --facet-title 'weak-incompatible=Weak private evidence,\nsocial evidence contradicts' \
  --facet-title 'strong-incompatible=Strong private evidence,\nsocial evidence contradicts' \
  --alignment-only \
  --hide-section-titles \
  --overlay-model-labels

FINAL_CHARTS_DERIVED_ONLY=1 python3 "$FIGURE_ROOT/code/make_flag_game_n_scaling_visuals.py"

rm -f "$FIGURE_ROOT/generated/flag_game_slot_figures_contact_sheet.png"
python3 "$FIGURE_ROOT/../../theory/make_social_circuit.py"
python3 "$FIGURE_ROOT/../../theory/make_figure.py"
python3 "$FIGURE_ROOT/code/validate_manifest.py"
