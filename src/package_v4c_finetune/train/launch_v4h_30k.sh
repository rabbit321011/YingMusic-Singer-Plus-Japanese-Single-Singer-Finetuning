#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${REMOTE_ROOT}/YingMusic-Singer-Plus"
RUNTIME_DIR="${REMOTE_ROOT}/TEMP/v4h_pul_runtime_20260729_v1"
OUTPUT_DIR="$PROJECT_DIR/ckpts/plus_ja_sft_v4h"

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "V4H output directory is not empty: $OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export H_PROJECT_DIR="$PROJECT_DIR"
export H_TRAIN_SCRIPT="$RUNTIME_DIR/train_plus_h.py"
export H_OUTPUT_DIR="$OUTPUT_DIR"
export H_MAX_STEPS="30000"
export H_SAVE_EVERY="2000"
export H_EVAL_EVERY="1000"
export H_LOG_EVERY="50"
export H_SMOKE_ASSERTIONS="1"

set +e
bash "$RUNTIME_DIR/run_sft_h_ddp.sh" phone_pul 2>&1 | tee "$OUTPUT_DIR/train.log"
status=${PIPESTATUS[0]}
set -e

printf '%s\n' "$status" > "$OUTPUT_DIR/exit_code.tmp"
mv "$OUTPUT_DIR/exit_code.tmp" "$OUTPUT_DIR/exit_code"
exit "$status"
