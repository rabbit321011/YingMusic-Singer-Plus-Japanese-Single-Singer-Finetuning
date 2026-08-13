#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${V4IJPH_PROJECT_DIR:-${REMOTE_ROOT}/YingMusic-Singer-Plus}"
H_DATA_DIR="${V4IJPH_H_DATA_DIR:-${REMOTE_ROOT}/TEMP/h_v1_full_20260728_v2/training}"
GAME_CACHE_DIR="${V4IJPH_GAME_CACHE_DIR:-${REMOTE_ROOT}/TEMP/v4ph_game_cache_full_20260730_v2}"
INS_CACHE_ROOT="${V4IJPH_INS_CACHE_ROOT:-${REMOTE_ROOT}/TEMP/v4ijph_ins_cache_v2}"
PYTHON="${V4IJPH_PYTHON:-${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/python}"
REPORT="${V4IJPH_SINGLE_PROBE_REPORT:-${REMOTE_ROOT}/TEMP/v4ijph_v2_single_probe.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_DIR"

exec "$PYTHON" "$SCRIPT_DIR/probe_v4ijph_single.py" \
    --source-v4iph-checkpoint "$PROJECT_DIR/ckpts/plus_ja_sft_v4iph/step_008000.pt" \
    --source-v4iph-sha256 "ddfd35941df398619349cc18597e3a63e4fb52019145476eb8f8b2c91dc15e10" \
    --game-cache-manifest "$GAME_CACHE_DIR/manifest.json" \
    --ins-train-cache "$INS_CACHE_ROOT/train" \
    --train-manifest "$H_DATA_DIR/train_h_training.json" \
    --train-manifest-sha256 "b2560dc62eace7e294bce800e99a2669973fd3bceed42003c8033e7d8cb24467" \
    --h-config-fingerprint "ca92066c0057e375e5d0485c54cb505ff8868a43e1b0efc528adcde6c1d11455" \
    --device cuda:0 \
    --report "$REPORT"
