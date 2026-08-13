#!/usr/bin/env bash
set -euo pipefail

PROJECT=${REMOTE_ROOT}/YingMusic-Singer-Plus
ROOT=${REMOTE_ROOT}/TEMP/v4m_m600d_listen_20260804
INPUT="$ROOT/input"
SCRIPTS="$INPUT/scripts"
DATASET="$INPUT/dataset"
ALIGNMENT="$INPUT/alignment"
RUNTIME="$INPUT/runtime"
PYTHON=${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/python3
VAE="$PROJECT/ckpts/stable_audio_2_0_vae_20hz_official.ckpt"
GAME_REPO=${REMOTE_ROOT}/TEMP/v4timetest_20260802/OpenVPI-GAME-full
GAME_DEPS=${REMOTE_ROOT}/.conda/envs/yingmusic_plus/lib/python3.10/site-packages
GAME_MODEL=${REMOTE_ROOT}/TEMP/v4timetest_20260802/GAME-1.0-medium/GAME-1.0-medium/model.pt
GAME_MANIFEST=${REMOTE_ROOT}/TEMP/v4ph_game_cache_full_20260730_v2/manifest.json
HIGHLR_DIR="$PROJECT/ckpts/plus_ja_sft_v4ph_30k_highlr"
PUBLISH="$ROOT/publish"
SMOKE="$ROOT/smoke"
GENERATED="$ROOT/generated"
LOG="$ROOT/generate.log"
EXIT="$ROOT/generate.exit"

mkdir -p "$SMOKE" "$GENERATED"
rm -f "$EXIT"
on_exit() {
  local rc=$?
  printf '%s\n' "$rc" > "$EXIT"
}
trap on_exit EXIT
exec > >(tee -a "$LOG") 2>&1

COMMON=(
  --dataset "$DATASET"
  --alignment-dir "$ALIGNMENT"
  --runtime "$RUNTIME"
  --singer-root "$PROJECT"
  --vae-ckpt "$VAE"
  --game-repo "$GAME_REPO"
  --game-deps "$GAME_DEPS"
  --game-model "$GAME_MODEL"
  --game-cache-manifest "$GAME_MANIFEST"
  --steps 32
  --seed 42
  --device cuda:0
  --resume
)

run_one() {
  local runner=$1
  local checkpoint=$2
  local label=$3
  local cfg=$4
  local output=$5
  local limit=${6:-}
  local extra=()
  if [[ -n "$limit" ]]; then
    extra=(--limit "$limit")
  fi
  printf '[generate] %s label=%s cfg=%s limit=%s\n' \
    "$(date -Is)" "$label" "$cfg" "${limit:-full}"
  local started
  started=$(date +%s)
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PROJECT:$SCRIPTS" \
    "$PYTHON" "$runner" \
      "${COMMON[@]}" \
      --checkpoint "$checkpoint" \
      --output-dir "$output" \
      --cfg "$cfg" \
      "${extra[@]}"
  printf '[elapsed] label=%s cfg=%s seconds=%s\n' \
    "$label" "$cfg" "$(( $(date +%s) - started ))"
}

declare -a RUNNERS=(
  "$SCRIPTS/v4ph_highlr_intermediate_eval_batch.py"
  "$SCRIPTS/v4ph_highlr_intermediate_eval_batch.py"
  "$SCRIPTS/v4m_m600d_eval_batch.py"
  "$SCRIPTS/v4m_m600d_eval_batch.py"
)
declare -a CHECKPOINTS=(
  "$HIGHLR_DIR/step_006000.pt"
  "$HIGHLR_DIR/step_012000.pt"
  "$PUBLISH/V4M_M600D_step_006000.pt"
  "$PUBLISH/V4M_M600D_step_012000.pt"
)
declare -a LABELS=(
  "HIGHLR_06K"
  "HIGHLR_12K"
  "M600D_06K"
  "M600D_12K"
)

printf '[smoke start] %s\n' "$(date -Is)"
for index in 0 1 2 3; do
  run_one \
    "${RUNNERS[$index]}" \
    "${CHECKPOINTS[$index]}" \
    "${LABELS[$index]}" \
    3.0 \
    "$SMOKE/${LABELS[$index]}_cfg3" \
    1
done
printf '[smoke complete] %s\n' "$(date -Is)"
"$PYTHON" "$PROJECT/package_v4c_finetune/infer/audit_v4m_m600d_listen_outputs.py" \
  --root "$SMOKE" \
  --dataset "$DATASET" \
  --expected-per-dir 1 \
  --report "$ROOT/smoke_audit.json" \
  --smoke

printf '[full start] %s\n' "$(date -Is)"
for index in 0 1 2 3; do
  for cfg in 3.0 1.0; do
    suffix=${cfg%.*}
    output="$GENERATED/${LABELS[$index]}_cfg${suffix}"
    run_one \
      "${RUNNERS[$index]}" \
      "${CHECKPOINTS[$index]}" \
      "${LABELS[$index]}" \
      "$cfg" \
      "$output"
    test "$(find "$output" -maxdepth 1 -type f -name '*.wav' | wc -l)" -eq 27
    test "$(find "$output/_placement" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 27
  done
done
"$PYTHON" "$PROJECT/package_v4c_finetune/infer/audit_v4m_m600d_listen_outputs.py" \
  --root "$GENERATED" \
  --dataset "$DATASET" \
  --expected-per-dir 27 \
  --report "$ROOT/full_audit.json"
printf '[full complete] %s\n' "$(date -Is)"
