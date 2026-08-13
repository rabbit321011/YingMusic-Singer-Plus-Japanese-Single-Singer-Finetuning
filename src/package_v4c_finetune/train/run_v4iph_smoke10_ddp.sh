#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export V4IPH_OUTPUT_DIR="${V4IPH_SMOKE_OUTPUT_DIR:-${REMOTE_ROOT}/TEMP/v4iph_smoke10}"

exec bash "$SCRIPT_DIR/run_v4iph_30k_ddp.sh" \
    --stop_after_step 10 \
    --save_every 2000 \
    --eval_every 1000 \
    --log_every 5 \
    "$@"
