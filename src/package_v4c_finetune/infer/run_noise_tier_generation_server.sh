#!/usr/bin/env bash
set -euo pipefail

PROJECT=${REMOTE_ROOT}/YingMusic-Singer-Plus
ROOT=${REMOTE_ROOT}/TEMP/noise_scorer_tiers_20260805
SCRIPTS="$ROOT/scripts"
PYTHON=${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/python3
V4PH_RUNNER=${REMOTE_ROOT}/TEMP/v4m_m600d_listen_20260804/input/scripts/v4ph_eval_batch.py
V4PH_RUNTIME="$ROOT/runtime_20260731"
V4PH_CHECKPOINT="$PROJECT/ckpts/plus_ja_sft_v4ph/step_030000_final.pt"
V4FG_CHECKPOINT="$PROJECT/ckpts/plus_ja_sft_v4fg/step_010000.pt"
OFFICIAL_VAE="$PROJECT/ckpts/stable_audio_2_0_vae_20hz_official.ckpt"
VAE_285K=${REMOTE_ROOT}/experiments/vae_full_official_300k_20260716/checkpoints/autoencoder_285k.ckpt
GAME_REPO=${REMOTE_ROOT}/TEMP/v4timetest_20260802/OpenVPI-GAME-full
GAME_DEPS=${REMOTE_ROOT}/.conda/envs/yingmusic_plus/lib/python3.10/site-packages
GAME_MODEL=${REMOTE_ROOT}/TEMP/v4timetest_20260802/GAME-1.0-medium/GAME-1.0-medium/model.pt
GAME_MANIFEST=${REMOTE_ROOT}/TEMP/v4ph_game_cache_full_20260730_v2/manifest.json
EXPECTED=500
LOG="$ROOT/generation_progress.log"
EXIT="$ROOT/generation.exit"

mkdir -p "$ROOT/generated/v4ph_cfg1" "$ROOT/generated/v4fg_cfg1"
rm -f "$EXIT"
on_exit() {
  local rc=$?
  printf '%s\n' "$rc" > "$EXIT"
}
trap on_exit EXIT

sha256sum \
  "$V4PH_CHECKPOINT" "$V4FG_CHECKPOINT" "$OFFICIAL_VAE" "$VAE_285K" \
  "$V4PH_RUNNER" "$SCRIPTS/v4fg_server_noise_tier_batch.py" \
  "$ROOT/v4ph_inputs/manifest.json" "$ROOT/v4fg_inputs/manifest.json" \
  > "$ROOT/generation_provenance.sha256"

run_monitored() {
  local label=$1
  local output=$2
  local raw_log=$3
  shift 3
  local started
  started=$(date +%s)
  printf '[start] %s label=%s expected=%s\n' "$(date -Is)" "$label" "$EXPECTED" | tee -a "$LOG"
  "$@" > "$raw_log" 2>&1 &
  local child=$!
  while kill -0 "$child" 2>/dev/null; do
    local now elapsed count eta
    now=$(date +%s)
    elapsed=$((now - started))
    count=$(find "$output" -maxdepth 1 -type f -name '*.wav' | wc -l)
    if (( count > 0 )); then
      eta=$((elapsed * (EXPECTED - count) / count))
    else
      eta=-1
    fi
    printf '[progress] %s label=%s completed=%s/%s elapsed=%ss eta=%ss\n' \
      "$(date -Is)" "$label" "$count" "$EXPECTED" "$elapsed" "$eta" | tee -a "$LOG"
    sleep 30
  done
  wait "$child"
  local count
  count=$(find "$output" -maxdepth 1 -type f -name '*.wav' | wc -l)
  test "$count" -eq "$EXPECTED"
  printf '[complete] %s label=%s completed=%s elapsed=%ss\n' \
    "$(date -Is)" "$label" "$count" "$(( $(date +%s) - started ))" | tee -a "$LOG"
}

test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 | tr -d ' ')" -lt 1024

run_monitored \
  V4PH "$ROOT/generated/v4ph_cfg1" "$ROOT/v4ph_generation_raw.log" \
  env CUDA_VISIBLE_DEVICES=4 \
      PYTHONPATH="$PROJECT:${REMOTE_ROOT}/TEMP/v4m_m600d_listen_20260804/input/scripts" \
      "$PYTHON" "$V4PH_RUNNER" \
        --dataset "$ROOT/v4ph_inputs" \
        --alignment-dir "$ROOT/v4ph_inputs/alignment" \
        --runtime "$V4PH_RUNTIME" \
        --singer-root "$PROJECT" \
        --checkpoint "$V4PH_CHECKPOINT" \
        --vae-ckpt "$OFFICIAL_VAE" \
        --game-repo "$GAME_REPO" \
        --game-deps "$GAME_DEPS" \
        --game-model "$GAME_MODEL" \
        --game-cache-manifest "$GAME_MANIFEST" \
        --steps 32 --cfg 1 --seed 42 --device cuda:0 --resume \
        --output-dir "$ROOT/generated/v4ph_cfg1"

run_monitored \
  V4fg "$ROOT/generated/v4fg_cfg1" "$ROOT/v4fg_generation_raw.log" \
  env CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PROJECT" \
      "$PYTHON" "$SCRIPTS/v4fg_server_noise_tier_batch.py" \
        --singer-root "$PROJECT" \
        --dataset "$ROOT/v4fg_inputs" \
        --output-dir "$ROOT/generated/v4fg_cfg1" \
        --checkpoint "$V4FG_CHECKPOINT" \
        --vae-ckpt "$VAE_285K" \
        --steps 32 --cfg 1 --seed 42 --device cuda:0 --resume

printf '[all complete] %s\n' "$(date -Is)" | tee -a "$LOG"
