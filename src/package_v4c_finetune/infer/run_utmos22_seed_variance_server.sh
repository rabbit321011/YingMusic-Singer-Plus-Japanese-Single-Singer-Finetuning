#!/usr/bin/env bash
set -euo pipefail

PROJECT=${REMOTE_ROOT}/YingMusic-Singer-Plus
ROOT=${REMOTE_ROOT}/TEMP/utmos22_seed_variance_20260806
INPUT=${REMOTE_ROOT}/TEMP/v4m_m600d_listen_20260804/input
SCORER_ROOT=${REMOTE_ROOT}/TEMP/noise_scorer_tiers_20260805
SCRIPTS="$ROOT/scripts"
PYTHON=${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/python3
V4PH_RUNNER="$INPUT/scripts/v4ph_eval_batch.py"
V4FG_RUNNER="$SCORER_ROOT/scripts/v4fg_server_noise_tier_batch.py"
V4PH_CHECKPOINT="$PROJECT/ckpts/plus_ja_sft_v4ph/step_030000_final.pt"
V4FG_CHECKPOINT="$PROJECT/ckpts/plus_ja_sft_v4fg/step_010000.pt"
OFFICIAL_VAE="$PROJECT/ckpts/stable_audio_2_0_vae_20hz_official.ckpt"
VAE_285K=${REMOTE_ROOT}/experiments/vae_full_official_300k_20260716/checkpoints/autoencoder_285k.ckpt
GAME_REPO=${REMOTE_ROOT}/TEMP/v4timetest_20260802/OpenVPI-GAME-full
GAME_DEPS=${REMOTE_ROOT}/.conda/envs/yingmusic_plus/lib/python3.10/site-packages
GAME_MODEL=${REMOTE_ROOT}/TEMP/v4timetest_20260802/GAME-1.0-medium/GAME-1.0-medium/model.pt
GAME_MANIFEST=${REMOTE_ROOT}/TEMP/v4ph_game_cache_full_20260730_v2/manifest.json
LOG="$ROOT/progress.log"
EXIT_FILE="$ROOT/run.exit"
CLOUD_DIR=/TEMP/utmos22_seed_variance_20260806
SEEDS=(42 43 44 45 46 47 48 49 50 51)

mkdir -p "$ROOT/generated/v4ph" "$ROOT/generated/v4fg"
rm -f "$EXIT_FILE"
finish() {
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" >"$EXIT_FILE"
  exit "$rc"
}
trap finish EXIT
exec > >(tee -a "$LOG") 2>&1

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 | tr -d ' ')
if (( used >= 1024 )); then
  printf '[error] GPU 4 is occupied: %s MiB\n' "$used"
  exit 1
fi

sha256sum \
  "$V4PH_CHECKPOINT" "$V4FG_CHECKPOINT" "$OFFICIAL_VAE" "$VAE_285K" \
  "$V4PH_RUNNER" "$V4FG_RUNNER" "$INPUT/dataset/manifest.json" \
  "$INPUT/alignment/manifest.json" "$SCRIPTS/analyze_utmos22_seed_variance.py" \
  >"$ROOT/provenance.sha256"

run_v4ph() {
  local seed=$1
  local output="$ROOT/generated/v4ph/seed_$(printf '%04d' "$seed")"
  local started
  started=$(date +%s)
  printf '[generate start] %s model=v4ph seed=%s\n' "$(date -Is)" "$seed"
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PROJECT:$INPUT/scripts" \
    "$PYTHON" "$V4PH_RUNNER" \
      --dataset "$INPUT/dataset" \
      --alignment-dir "$INPUT/alignment" \
      --runtime "$SCORER_ROOT/runtime_20260731" \
      --singer-root "$PROJECT" \
      --checkpoint "$V4PH_CHECKPOINT" \
      --vae-ckpt "$OFFICIAL_VAE" \
      --game-repo "$GAME_REPO" \
      --game-deps "$GAME_DEPS" \
      --game-model "$GAME_MODEL" \
      --game-cache-manifest "$GAME_MANIFEST" \
      --steps 32 --cfg 1 --seed "$seed" --device cuda:0 --resume \
      --output-dir "$output"
  test "$(find "$output" -maxdepth 1 -type f -name '*.wav' | wc -l)" -eq 27
  test "$(find "$output/_placement" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 27
  printf '[generate complete] %s model=v4ph seed=%s elapsed=%ss\n' \
    "$(date -Is)" "$seed" "$(( $(date +%s) - started ))"
}

run_v4fg() {
  local seed=$1
  local output="$ROOT/generated/v4fg/seed_$(printf '%04d' "$seed")"
  local started
  started=$(date +%s)
  printf '[generate start] %s model=v4fg seed=%s\n' "$(date -Is)" "$seed"
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PROJECT" \
    "$PYTHON" "$V4FG_RUNNER" \
      --singer-root "$PROJECT" \
      --dataset "$INPUT/dataset" \
      --output-dir "$output" \
      --checkpoint "$V4FG_CHECKPOINT" \
      --vae-ckpt "$VAE_285K" \
      --steps 32 --cfg 1 --seed "$seed" --device cuda:0 --resume
  test "$(find "$output" -maxdepth 1 -type f -name '*.wav' | wc -l)" -eq 27
  printf '[generate complete] %s model=v4fg seed=%s elapsed=%ss\n' \
    "$(date -Is)" "$seed" "$(( $(date +%s) - started ))"
}

printf '[experiment start] %s seeds=%s\n' "$(date -Is)" "${SEEDS[*]}"
for seed in "${SEEDS[@]}"; do
  run_v4ph "$seed"
done
for seed in "${SEEDS[@]}"; do
  run_v4fg "$seed"
done

printf '[score start] %s\n' "$(date -Is)"
CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PROJECT" \
  "$SCORER_ROOT/venv_scorers/bin/python" "$SCRIPTS/analyze_utmos22_seed_variance.py" \
    --root "$ROOT" --scorer-root "$SCORER_ROOT" --device cuda:0
printf '[score complete] %s\n' "$(date -Is)"

if ! ${REMOTE_ROOT}/bin/aliyunpan ls "$CLOUD_DIR" >/dev/null 2>&1; then
  ${REMOTE_ROOT}/bin/aliyunpan mkdir "$CLOUD_DIR"
fi
${REMOTE_ROOT}/bin/aliyunpan upload -p 2 --retry 3 --timeout 90 \
  "$ROOT/summary.json" \
  "$ROOT/per_task_stats.csv" \
  "$ROOT/utmos22_scores.csv" \
  "$ROOT/scorer_repeatability.csv" \
  "$ROOT/canonical_manifest.csv" \
  "$ROOT/provenance.sha256" \
  "$ROOT/listen_extremes.zip" \
  "$CLOUD_DIR"
printf '[all complete] %s cloud=%s\n' "$(date -Is)" "$CLOUD_DIR"
