#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runpod_cache_env.sh
source "$ROOT/scripts/runpod_cache_env.sh"
cd "$ROOT"

MODEL="${MODEL:-llava-hf/llava-v1.6-mistral-7b-hf}"
OUT="${OUT:-runs/llava_memory_conflict_probe}"
M_VALUES="${M_VALUES:-3}"
TRUTH_COUNTRIES="${TRUTH_COUNTRIES:-Czech Republic,Peru,Guinea,Bahamas}"
CROP_CONDITIONS="${CROP_CONDITIONS:-diagnostic_true,ambiguous_true}"
LURE_RELATIONS="${LURE_RELATIONS:-compatible,incompatible}"
MEMORY_COUNTS="${MEMORY_COUNTS:-0,1,2,3,4,5,6,7,8}"
REPLICATES="${REPLICATES:-3}"
SEED="${SEED:-0}"
MAX_TOKENS="${MAX_TOKENS:-200}"
TEMPERATURE="${TEMPERATURE:-0.2}"
ACTIVATION_CAPTURE="${ACTIVATION_CAPTURE:-0}"
ACTIVATION_LAYERS="${ACTIVATION_LAYERS:-}"
ACTIVATION_FEATURE="${ACTIVATION_FEATURE:-last_prompt_token}"

export NND_TRANSFORMERS_VLM_FAMILY="${NND_TRANSFORMERS_VLM_FAMILY:-llava_next}"
export NND_TRANSFORMERS_DTYPE="${NND_TRANSFORMERS_DTYPE:-bfloat16}"
export NND_TRANSFORMERS_DEVICE_MAP="${NND_TRANSFORMERS_DEVICE_MAP:-auto}"
export NND_TRANSFORMERS_ATTN_IMPLEMENTATION="${NND_TRANSFORMERS_ATTN_IMPLEMENTATION:-auto}"

args=(
  python scripts/run_flag_memory_conflict_probe.py
  --backend transformers_vlm
  --model "$MODEL"
  --out "$OUT"
  --memory-counts "$MEMORY_COUNTS"
  --replicates "$REPLICATES"
  --seed "$SEED"
  --temperature "$TEMPERATURE"
  --max-tokens "$MAX_TOKENS"
)

IFS=',' read -r -a m_values <<< "$M_VALUES"
for value in "${m_values[@]}"; do
  [[ -n "$value" ]] && args+=(--m "$value")
done

IFS=',' read -r -a countries <<< "$TRUTH_COUNTRIES"
for value in "${countries[@]}"; do
  [[ -n "$value" ]] && args+=(--truth-country "$value")
done

IFS=',' read -r -a crop_conditions <<< "$CROP_CONDITIONS"
for value in "${crop_conditions[@]}"; do
  [[ -n "$value" ]] && args+=(--crop-condition "$value")
done

IFS=',' read -r -a lure_relations <<< "$LURE_RELATIONS"
for value in "${lure_relations[@]}"; do
  [[ -n "$value" ]] && args+=(--lure-relation "$value")
done

if [[ "$ACTIVATION_CAPTURE" == "1" ]]; then
  args+=(--activation-capture --activation-scope all_probes)
  if [[ -n "$ACTIVATION_LAYERS" ]]; then
    args+=(--activation-layers "$ACTIVATION_LAYERS")
  fi
fi

echo "LLaVA memory-conflict probe"
echo "  model: $MODEL"
echo "  out:   $OUT"
echo "  m:     $M_VALUES"
echo "  reps:  $REPLICATES"
echo "  activation capture: $ACTIVATION_CAPTURE"

"${args[@]}"

if [[ "$ACTIVATION_CAPTURE" == "1" ]]; then
  python scripts/analyze_memory_conflict_activations.py \
    --runs "$OUT" \
    --feature "$ACTIVATION_FEATURE" \
    --out "$OUT/activation_concepts_$ACTIVATION_FEATURE"
fi

echo
echo "Memory-conflict probe complete."
echo "Key outputs:"
echo "  $OUT/results.csv"
echo "  $OUT/summary_by_count.csv"
echo "  $OUT/threshold_summary.csv"
if [[ "$ACTIVATION_CAPTURE" == "1" ]]; then
  echo "  $OUT/activations/index.jsonl"
  echo "  $OUT/activation_concepts_$ACTIVATION_FEATURE/concept_separation_by_layer.csv"
fi
