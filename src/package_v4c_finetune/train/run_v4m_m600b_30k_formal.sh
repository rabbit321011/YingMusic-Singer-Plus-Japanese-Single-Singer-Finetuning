#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${V4M_PROJECT_DIR:-${REMOTE_ROOT}/YingMusic-Singer-Plus}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export V4M_TRANSITION="${V4M_TRANSITION:-$PROJECT_DIR/ckpts/v4m_m600b/m600b_transition.pt}"
export V4M_OUTPUT_DIR="${V4M_OUTPUT_DIR:-$PROJECT_DIR/ckpts/v4m_m600b_30k_20260805}"

exec "$SCRIPT_DIR/run_v4m_m600b_fsdp.sh" \
    --stop_after_step 30000 \
    --save_every 1000 \
    --log_every 10 \
    --record_every 10 \
    --empty_cache_every 100 \
    --min_free_gib 3.0 \
    --trace_limit 100 \
    "$@"
