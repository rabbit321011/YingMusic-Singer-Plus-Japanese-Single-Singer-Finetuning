#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${V4PH_PROJECT_DIR:-${REMOTE_ROOT}/YingMusic-Singer-Plus}"
H_DATA_DIR="${V4PH_H_DATA_DIR:-${REMOTE_ROOT}/TEMP/h_v1_full_20260728_v2/training}"
CACHE_DIR="${V4PH_CACHE_DIR:-${REMOTE_ROOT}/TEMP/v4ph_game_cache_full_20260730_v2}"
TRAIN_SCRIPT="${V4PH_TRAIN_SCRIPT:-$SCRIPT_DIR/train_v4ph.py}"
TORCHRUN="${V4PH_TORCHRUN:-${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/torchrun}"
OUTPUT_DIR="${V4PH_OUTPUT_DIR:-$PROJECT_DIR/ckpts/v4ph_phase_a}"
NPROC="${V4PH_NPROC:-4}"

TRAIN_MANIFEST="$H_DATA_DIR/train_h_training.json"
EVAL_MANIFEST="$H_DATA_DIR/test_h_training.json"
TRAIN_SHA="b2560dc62eace7e294bce800e99a2669973fd3bceed42003c8033e7d8cb24467"
EVAL_SHA="befbf1341292f0c26707be6771178b381689c0de83682853e700bd9e1f5f38cc"
H_CONFIG="ca92066c0057e375e5d0485c54cb505ff8868a43e1b0efc528adcde6c1d11455"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================"
echo "V4PH phase A: Official base + H/PUL + GAME P"
echo "Trainable: P embedding only"
echo "Loss: FlowB + 0.7 * GAME-compatible CKA"
echo "LR: constant 1e-4; steps: 500"
echo "GPUs: $CUDA_VISIBLE_DEVICES (world=$NPROC)"
echo "GAME cache: $CACHE_DIR/manifest.json"
echo "Output: $OUTPUT_DIR"
echo "========================================"

cd "$PROJECT_DIR"

exec "$TORCHRUN" --nproc_per_node="$NPROC" "$TRAIN_SCRIPT" \
    --phase p_only \
    --train_manifest "$TRAIN_MANIFEST" \
    --eval_manifest "$EVAL_MANIFEST" \
    --train_manifest_sha256 "$TRAIN_SHA" \
    --eval_manifest_sha256 "$EVAL_SHA" \
    --h_config_fingerprint "$H_CONFIG" \
    --game_cache_manifest "$CACHE_DIR/manifest.json" \
    --placement_mode phone_pul \
    --output_dir "$OUTPUT_DIR" \
    --ckpt_path ckpts/YingMusicSinger_model.pt \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 1e-4 \
    --warmup_steps 0 \
    --hold_steps 500 \
    --max_steps 500 \
    --save_every 100 \
    --eval_every 100 \
    --log_every 25 \
    --max_duration 30 \
    --num_workers 0 \
    --cka_weight 0.7 \
    --drop_text 0 \
    --flow_b_weight 2.0 \
    --seed 42 \
    --eval_seed 1042 \
    --expected_world_size "$NPROC" \
    --smoke_assertions \
    "$@"
