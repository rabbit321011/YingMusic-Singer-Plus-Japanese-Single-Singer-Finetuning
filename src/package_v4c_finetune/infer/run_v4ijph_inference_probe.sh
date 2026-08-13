#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${V4IJPH_PROJECT_DIR:-${REMOTE_ROOT}/YingMusic-Singer-Plus}"
CHECKPOINT="${V4IJPH_INFERENCE_CHECKPOINT:?set V4IJPH_INFERENCE_CHECKPOINT}"
REPORT="${V4IJPH_INFERENCE_REPORT:-${REMOTE_ROOT}/TEMP/v4ijph_v2_inference_probe.json}"
PYTHON="${V4IJPH_PYTHON:-${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_DIR"

exec "$PYTHON" "$SCRIPT_DIR/probe_v4ijph_inference.py" \
    --checkpoint "$CHECKPOINT" \
    --game-cache-manifest ${REMOTE_ROOT}/TEMP/v4ph_game_cache_full_20260730_v2/manifest.json \
    --eval-manifest ${REMOTE_ROOT}/TEMP/h_v1_full_20260728_v2/training/test_h_training.json \
    --eval-manifest-sha256 befbf1341292f0c26707be6771178b381689c0de83682853e700bd9e1f5f38cc \
    --h-config-fingerprint ca92066c0057e375e5d0485c54cb505ff8868a43e1b0efc528adcde6c1d11455 \
    --ins-eval-cache ${REMOTE_ROOT}/TEMP/v4ijph_ins_cache_v2/eval \
    --device cuda:0 \
    --steps 2 \
    --cfg-strength 1 \
    --seed 20260801 \
    --report "$REPORT"
