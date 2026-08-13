#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export V4IJPH_OUTPUT_DIR="${V4IJPH_SMOKE_OUTPUT_DIR:-${REMOTE_ROOT}/TEMP/v4ijph_v2_active_smoke8010}"

exec bash "$SCRIPT_DIR/run_v4ijph_30k_ddp.sh" \
    --stop_after_step 8010 \
    --save_every 2000 \
    --eval_every 1000 \
    --log_every 5 \
    "$@"
