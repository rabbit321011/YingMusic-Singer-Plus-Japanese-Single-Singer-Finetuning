#!/usr/bin/env bash
set -euo pipefail

ROOT=${REMOTE_ROOT}/TEMP/noise_scorer_tiers_20260805
WEIGHTS="$ROOT/scorer_weights"
S3PRL_CACHE=${REMOTE_ROOT}/.cache/s3prl/download
UTMOSV2_CACHE=${REMOTE_ROOT}/.cache/utmosv2/models/fusion_stage3
EXIT_FILE="$ROOT/scorer_weight_download.exit"

rm -f "$EXIT_FILE"
finish() {
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" >"$EXIT_FILE"
  exit "$rc"
}
trap finish EXIT

mkdir -p "$WEIGHTS" "$S3PRL_CACHE" "$UTMOSV2_CACHE"

fetch() {
  local destination=$1
  local url=$2
  if [[ -s "$destination" ]]; then
    printf '[skip] %s (%s bytes)\n' "$destination" "$(stat -c %s "$destination")"
    return
  fi
  printf '[download] %s\n' "$destination"
  wget --continue --progress=dot:giga --output-document="$destination.part" "$url"
  mv "$destination.part" "$destination"
  printf '[complete] %s (%s bytes)\n' "$destination" "$(stat -c %s "$destination")"
}

fetch \
  "$S3PRL_CACHE/aa064e275fe0123a0e1b515f2341bbe4408368510d91d0a6816f2822a6e5acdd.wav2vec_small.pt" \
  "https://hf-mirror.com/s3prl/converted_ckpts/resolve/main/wav2vec_small.pt"
fetch \
  "$S3PRL_CACHE/70a1c0d7bd4d235fe73bc359f70cc91a165d2405f6608fb01c504b8ba1244ac6.wav2vec_vox_new.pt" \
  "https://hf-mirror.com/s3prl/converted_ckpts/resolve/main/wav2vec_vox_new.pt"
fetch \
  "$WEIGHTS/singmos_v1.pt" \
  "https://github.com/South-Twilight/SingMOS/releases/download/ckpt_s3prl/ft_wav2vec2_base_960_23steps.pt"
fetch \
  "$WEIGHTS/singmos_pro.pth" \
  "https://github.com/South-Twilight/SingMOS/releases/download/ckpt_v3/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth"
fetch \
  "$WEIGHTS/utmos22_strong_step7459_v1.pt" \
  "https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt"
fetch \
  "$UTMOSV2_CACHE/fold0_s42_best_model.pth" \
  "https://hf-mirror.com/sarulab-speech/UTMOSv2/resolve/main/fold0_s42_best_model.pth"

sha256sum \
  "$S3PRL_CACHE/aa064e275fe0123a0e1b515f2341bbe4408368510d91d0a6816f2822a6e5acdd.wav2vec_small.pt" \
  "$S3PRL_CACHE/70a1c0d7bd4d235fe73bc359f70cc91a165d2405f6608fb01c504b8ba1244ac6.wav2vec_vox_new.pt" \
  "$WEIGHTS/singmos_v1.pt" \
  "$WEIGHTS/singmos_pro.pth" \
  "$WEIGHTS/utmos22_strong_step7459_v1.pt" \
  "$UTMOSV2_CACHE/fold0_s42_best_model.pth" \
  >"$ROOT/scorer_weight_sha256.txt"

printf '[all complete]\n'
