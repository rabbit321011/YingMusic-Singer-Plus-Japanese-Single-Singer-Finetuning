#!/usr/bin/env bash

set -euo pipefail

PROJECT="${REMOTE_ROOT}/YingMusic-Singer-Plus"
SOURCE="$PROJECT/ckpts/plus_ja_sft_v4ph/step_030000_final.pt"
SIDECAR="$SOURCE.sha256"
EXPECTED_SHA="80cf7251c19a0d803a7bb0ce9e363fd1700c102d10c52de93a486180aa73372c"
TARGET="/TEMP/v4ph_30k"
LOG="${REMOTE_ROOT}/TEMP/v4ph_30k_cloud_upload.log"
ALIYUNPAN="${REMOTE_ROOT}/bin/aliyunpan"

actual_sha="$(sha256sum "$SOURCE" | cut -d' ' -f1)"
if [[ "$actual_sha" != "$EXPECTED_SHA" ]]; then
    echo "source SHA256 mismatch: $actual_sha != $EXPECTED_SHA" >&2
    exit 1
fi
printf '%s  %s\n' "$EXPECTED_SHA" "$(basename "$SOURCE")" > "$SIDECAR"

"$ALIYUNPAN" mkdir "$TARGET" >/dev/null 2>&1 || true
echo "START $(date -Is)" | tee "$LOG"
"$ALIYUNPAN" upload -bs 30720 "$SOURCE" "$SIDECAR" "$TARGET" 2>&1 | tee -a "$LOG"
echo "EXIT ${PIPESTATUS[0]} $(date -Is)" | tee -a "$LOG"
"$ALIYUNPAN" ls "$TARGET" | tee -a "$LOG"
