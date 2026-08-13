#!/usr/bin/env bash
set -euo pipefail

PROJECT=${REMOTE_ROOT}/YingMusic-Singer-Plus
ROOT=${REMOTE_ROOT}/TEMP/v4m_m600d_listen_20260804
PYTHON=${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/python3
BUILDER="$PROJECT/package_v4c_finetune/infer/build_v4m_m600d_listen_package.py"
PACKAGE="$ROOT/package"
BLIND="$PACKAGE/V4M_M600D_same_step_blind_20260804.zip"
TECHNICAL="$PACKAGE/V4M_M600D_same_step_technical_20260804.zip"
REPORT="$PACKAGE/package_report.json"
ALIYUNPAN=${REMOTE_ROOT}/bin/aliyunpan
CLOUD=/TEMP/v4m_m600d_listen_20260804
LOG="$ROOT/package_upload.log"
EXIT="$ROOT/package_upload.exit"

rm -f "$EXIT"
on_exit() {
  local rc=$?
  printf '%s\n' "$rc" > "$EXIT"
}
trap on_exit EXIT
exec > >(tee -a "$LOG") 2>&1

test "$(cat "$ROOT/generate.exit")" = 0
printf '[package start] %s\n' "$(date -Is)"
"$PYTHON" "$BUILDER" --root "$ROOT"
"$PYTHON" -c "import json,pathlib; p=pathlib.Path('$REPORT'); x=json.loads(p.read_text()); assert x['status']=='ok' and x['wav_count']==216 and x['score_rows']==216; print('[package audit ok]',p)"

"$ALIYUNPAN" mkdir "$CLOUD" >/dev/null 2>&1 || true
printf '[upload start] %s\n' "$(date -Is)"
"$ALIYUNPAN" upload -bs 30720 "$BLIND" "$TECHNICAL" "$REPORT" "$CLOUD"
"$ALIYUNPAN" ls "$CLOUD"
printf '[package/upload complete] %s\n' "$(date -Is)"
