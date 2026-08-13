#!/usr/bin/env bash

set -uo pipefail

RUNTIME=${REMOTE_ROOT}/TEMP/v4hg_runtime_20260730_v1
OUTPUT=${REMOTE_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_sft_v4hg_10k
LOG=${REMOTE_ROOT}/TEMP/v4hg_10k_20260730_v1.log
EXIT_CODE=${REMOTE_ROOT}/TEMP/v4hg_10k_20260730_v1.exit_code

for path in "$OUTPUT" "$LOG" "$EXIT_CODE"; do
    if [[ -e "$path" ]]; then
        echo "Refusing to overwrite existing formal artifact: $path" >&2
        exit 2
    fi
done

for gpu in 0 1 2 3; do
    used="$(
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i "$gpu" | tr -d ' '
    )"
    if (( used > 500 )); then
        echo "GPU $gpu is busy: ${used}MiB" >&2
        exit 2
    fi
done

cd "$RUNTIME" || exit 2
CUDA_VISIBLE_DEVICES=0,1,2,3 \
H_OUTPUT_DIR="$OUTPUT" \
H_SMOKE_ASSERTIONS=1 \
    ./run_sft_v4hg_10k.sh 2>&1 | tee "$LOG"
status="${PIPESTATUS[0]}"
printf '%s\n' "$status" > "$EXIT_CODE"
exit "$status"
