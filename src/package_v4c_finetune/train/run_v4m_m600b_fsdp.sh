#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${V4M_PROJECT_DIR:-${REMOTE_ROOT}/YingMusic-Singer-Plus}"
H_DATA_DIR="${V4M_H_DATA_DIR:-${REMOTE_ROOT}/TEMP/h_v1_full_20260728_v2/training}"
CACHE_DIR="${V4M_CACHE_DIR:-${REMOTE_ROOT}/TEMP/v4ph_game_cache_full_20260730_v2}"
TRANSITION="${V4M_TRANSITION:-${REMOTE_ROOT}/YingMusic-Singer-Plus/ckpts/v4m_m600b/m600b_transition.pt}"
OUTPUT_DIR="${V4M_OUTPUT_DIR:-${REMOTE_ROOT}/TEMP/v4m_m600b_fsdp}"
TORCHRUN="${V4M_TORCHRUN:-${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/torchrun}"
NPROC="${V4M_NPROC:-4}"

TRAIN_MANIFEST="$H_DATA_DIR/train_h_training.json"
TRAIN_SHA="b2560dc62eace7e294bce800e99a2669973fd3bceed42003c8033e7d8cb24467"
H_CONFIG="ca92066c0057e375e5d0485c54cb505ff8868a43e1b0efc528adcde6c1d11455"
TRANSITION_SHA="c152b15ad2142b5233d93f05914682aad752d023a56b3bf4627b7f3edd8d1e6f"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT_DIR"

exec "$TORCHRUN" --nproc_per_node="$NPROC" \
    "$SCRIPT_DIR/train_v4m_m600b_fsdp.py" \
    --transition "$TRANSITION" \
    --transition_sha256 "$TRANSITION_SHA" \
    --train_manifest "$TRAIN_MANIFEST" \
    --train_manifest_sha256 "$TRAIN_SHA" \
    --h_config_fingerprint "$H_CONFIG" \
    --game_cache_manifest "$CACHE_DIR/manifest.json" \
    --output_dir "$OUTPUT_DIR" \
    --grad_accum 4 \
    --lr 1.4e-5 \
    --warmup_steps 500 \
    --hold_steps 23500 \
    --max_steps 30000 \
    --max_duration 30 \
    --flow_b_weight 2.0 \
    --cka_weight 0.7 \
    --drop_text 0.15 \
    --seed 42 \
    --expected_world_size "$NPROC" \
    "$@"
