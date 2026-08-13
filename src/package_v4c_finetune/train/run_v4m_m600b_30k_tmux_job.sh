#!/usr/bin/env bash

set -uo pipefail

PROJECT_DIR="${V4M_PROJECT_DIR:-${REMOTE_ROOT}/YingMusic-Singer-Plus}"
LOG="${V4M_LOG:-$PROJECT_DIR/ckpts/v4m_m600b_30k_20260805.log}"
EXIT_FILE="${V4M_EXIT_FILE:-$PROJECT_DIR/ckpts/v4m_m600b_30k_20260805.exit}"

if [[ -e "$LOG" || -e "$EXIT_FILE" ]]; then
    printf 'refusing to overwrite existing log or exit file\n' >&2
    exit 2
fi

"$PROJECT_DIR/package_v4c_finetune/train/run_v4m_m600b_30k_formal.sh" \
    >"$LOG" 2>&1
status=$?
printf '%s\n' "$status" >"$EXIT_FILE"
exit "$status"
