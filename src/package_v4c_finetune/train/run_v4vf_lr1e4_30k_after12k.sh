#!/bin/bash
# Wait for the independent 1e-4 12k run, then start a new random-init 30k run.
# The output directory is intentionally different from the 12k run.

set -euo pipefail

cd "$(dirname "$0")"

OUTPUT_DIR="ckpts/plus_ja_sft_v4vf_lr1e4_30k"
LOG_PATH="train_v4vf_lr1e4_30k.log"
UPSTREAM_SESSION="v4vf_lr1e4_12k"

echo "[V4vf scheduler] Waiting for upstream session $UPSTREAM_SESSION to finish"
while tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; do
    sleep 60
done

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "[V4vf scheduler] Refusing to overwrite existing output: $OUTPUT_DIR" >&2
    exit 2
fi

echo "[V4vf scheduler] Upstream finished; waiting for GPUs 0,1,2,3"
while true; do
    busy=0
    while IFS=, read -r index memory; do
        memory=${memory// MiB/}
        if (( memory > 500 )); then
            busy=1
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F', ' '$1 <= 3 {print $1 "," $2}')
    if (( busy == 0 )); then
        break
    fi
    sleep 60
done

echo "[V4vf scheduler] Starting new random-init global-step-30k run"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
bash run_sft_v4vf_random_init_ddp.sh \
    --output_dir "$OUTPUT_DIR" \
    --max_steps 30000 \
    --lr 1e-4 \
    2>&1 | tee "$LOG_PATH"
