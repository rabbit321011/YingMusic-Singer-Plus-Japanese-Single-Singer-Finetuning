#!/usr/bin/env bash
set -euo pipefail

ROOT=${REMOTE_ROOT}/TEMP/noise_scorer_tiers_20260805
PYTHON="$ROOT/venv_scorers/bin/python"
BUILDER="$ROOT/scripts/build_noise_tier_packages.py"
PROGRESS="$ROOT/finalize_progress.log"
EXIT_FILE="$ROOT/finalize.exit"
CLOUD_DIR=/TEMP/noise_scorer_tiers_20260805

rm -f "$EXIT_FILE"
: >"$PROGRESS"
finish() {
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" >"$EXIT_FILE"
  exit "$rc"
}
trap finish EXIT

score_count() {
  local name=$1
  if [[ ! -s "$ROOT/scores/$name.csv" ]]; then
    printf '0'
    return
  fi
  local lines
  lines=$(wc -l <"$ROOT/scores/$name.csv")
  printf '%s' "$((lines - 1))"
}

started=$(date +%s)
while true; do
  gpu=running
  cpu=running
  mosnet=running
  [[ -f "$ROOT/gpu_scoring.exit" ]] && gpu=$(cat "$ROOT/gpu_scoring.exit")
  [[ -f "$ROOT/cpu_scoring.exit" ]] && cpu=$(cat "$ROOT/cpu_scoring.exit")
  [[ -f "$ROOT/mosnet_scoring.exit" ]] && mosnet=$(cat "$ROOT/mosnet_scoring.exit")
  printf '[wait] %s elapsed=%ss gpu=%s cpu=%s mosnet=%s rows(nisqa/dnsmos/mosnet)=%s/%s/%s\n' \
    "$(date --iso-8601=seconds)" "$(( $(date +%s) - started ))" \
    "$gpu" "$cpu" "$mosnet" \
    "$(score_count nisqa)" "$(score_count dnsmos)" "$(score_count mosnet)" \
    | tee -a "$PROGRESS"
  if [[ -f "$ROOT/gpu_scoring.exit" && -f "$ROOT/cpu_scoring.exit" && -f "$ROOT/mosnet_scoring.exit" ]]; then
    break
  fi
  sleep 60
done

for file in gpu_scoring.exit cpu_scoring.exit mosnet_scoring.exit; do
  rc=$(cat "$ROOT/$file")
  if [[ "$rc" != 0 ]]; then
    printf '[error] %s=%s\n' "$file" "$rc" | tee -a "$PROGRESS"
    exit "$rc"
  fi
done

printf '[build start] %s\n' "$(date --iso-8601=seconds)" | tee -a "$PROGRESS"
"$PYTHON" "$BUILDER" --root "$ROOT" --overwrite \
  2>&1 | tee "$ROOT/package_build.log"
printf '[build complete] %s\n' "$(date --iso-8601=seconds)" | tee -a "$PROGRESS"

if ! ${REMOTE_ROOT}/bin/aliyunpan ls "$CLOUD_DIR" >/dev/null 2>&1; then
  ${REMOTE_ROOT}/bin/aliyunpan mkdir "$CLOUD_DIR"
fi
printf '[upload start] %s cloud=%s\n' "$(date --iso-8601=seconds)" "$CLOUD_DIR" \
  | tee -a "$PROGRESS"
${REMOTE_ROOT}/bin/aliyunpan upload -p 2 --retry 3 --timeout 90 \
  "$ROOT/package_archives" \
  "$ROOT/all_scores.csv" \
  "$ROOT/package_audit.json" \
  "$ROOT/canonical_audio_audit.json" \
  "$ROOT/scorer_weight_sha256.txt" \
  "$CLOUD_DIR" \
  2>&1 | tee "$ROOT/final_upload.log"
printf '[upload complete] %s cloud=%s\n' "$(date --iso-8601=seconds)" "$CLOUD_DIR" \
  | tee -a "$PROGRESS"
