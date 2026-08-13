#!/usr/bin/env bash
set -uo pipefail

ROOT=${REMOTE_ROOT}/TEMP/noise_scorer_tiers_20260805
PYTHON="$ROOT/venv_scorers/bin/python"
SCRIPT="$ROOT/scripts/prepare_noise_tier_scoring_audio.py"
LOG="$ROOT/normalize.log"
EXIT_FILE="$ROOT/normalize.exit"

rm -f "$EXIT_FILE"
"$PYTHON" "$SCRIPT" --root "$ROOT" >"$LOG" 2>&1
rc=$?
printf '%s\n' "$rc" >"$EXIT_FILE"
exit "$rc"
