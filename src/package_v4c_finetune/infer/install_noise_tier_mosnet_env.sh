#!/usr/bin/env bash
set -euo pipefail

ROOT=${REMOTE_ROOT}/TEMP/noise_scorer_tiers_20260805
VENV="$ROOT/venv_mosnet"
EXIT_FILE="$ROOT/mosnet_env_install.exit"

rm -f "$EXIT_FILE"
finish() {
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" >"$EXIT_FILE"
  exit "$rc"
}
trap finish EXIT

if [[ ! -x "$VENV/bin/python" ]]; then
  ${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/python -m venv \
    --system-site-packages "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade "pip<26"
"$VENV/bin/python" -m pip install "tensorflow-cpu==2.15.1"
"$VENV/bin/python" - <<'PY'
import librosa
import scipy
import tensorflow as tf
print("tensorflow", tf.__version__)
print("librosa", librosa.__version__)
print("scipy", scipy.__version__)
PY
