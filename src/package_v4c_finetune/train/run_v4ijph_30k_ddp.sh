#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${V4IJPH_PROJECT_DIR:-${REMOTE_ROOT}/YingMusic-Singer-Plus}"
H_DATA_DIR="${V4IJPH_H_DATA_DIR:-${REMOTE_ROOT}/TEMP/h_v1_full_20260728_v2/training}"
GAME_CACHE_DIR="${V4IJPH_GAME_CACHE_DIR:-${REMOTE_ROOT}/TEMP/v4ph_game_cache_full_20260730_v2}"
INS_CACHE_ROOT="${V4IJPH_INS_CACHE_ROOT:-${REMOTE_ROOT}/TEMP/v4ijph_ins_cache_v2}"
TRAIN_SCRIPT="${V4IJPH_TRAIN_SCRIPT:-$SCRIPT_DIR/train_v4ijph.py}"
TORCHRUN="${V4IJPH_TORCHRUN:-${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/torchrun}"
OUTPUT_DIR="${V4IJPH_OUTPUT_DIR:-$PROJECT_DIR/ckpts/plus_ja_sft_v4ijph_v2}"
NPROC="${V4IJPH_NPROC:-4}"

TRAIN_MANIFEST="$H_DATA_DIR/train_h_training.json"
EVAL_MANIFEST="$H_DATA_DIR/test_h_training.json"
TRAIN_SHA="b2560dc62eace7e294bce800e99a2669973fd3bceed42003c8033e7d8cb24467"
EVAL_SHA="befbf1341292f0c26707be6771178b381689c0de83682853e700bd9e1f5f38cc"
H_CONFIG="ca92066c0057e375e5d0485c54cb505ff8868a43e1b0efc528adcde6c1d11455"
SOURCE_V4IPH="$PROJECT_DIR/ckpts/plus_ja_sft_v4iph/step_008000.pt"
SOURCE_V4IPH_SHA="ddfd35941df398619349cc18597e3a63e4fb52019145476eb8f8b2c91dc15e10"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "============================================================"
echo "V4IjPH v2: exact V4IPH@8k transition + full-timeline INS"
echo "Source: $SOURCE_V4IPH"
echo "INS cache: $INS_CACHE_ROOT"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

cd "$PROJECT_DIR"
exec "$TORCHRUN" --nproc_per_node="$NPROC" "$TRAIN_SCRIPT" \
    --source_v4iph_checkpoint "$SOURCE_V4IPH" \
    --source_v4iph_sha256 "$SOURCE_V4IPH_SHA" \
    --train_manifest "$TRAIN_MANIFEST" \
    --eval_manifest "$EVAL_MANIFEST" \
    --train_manifest_sha256 "$TRAIN_SHA" \
    --eval_manifest_sha256 "$EVAL_SHA" \
    --h_config_fingerprint "$H_CONFIG" \
    --game_cache_manifest "$GAME_CACHE_DIR/manifest.json" \
    --ins_train_cache "$INS_CACHE_ROOT/train" \
    --ins_eval_cache "$INS_CACHE_ROOT/eval" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 1.4e-5 \
    --warmup_steps 500 \
    --hold_steps 12000 \
    --max_steps 30000 \
    --save_every 2000 \
    --eval_every 1000 \
    --log_every 50 \
    --max_duration 30 \
    --num_workers 0 \
    --cka_weight 0.7 \
    --drop_text 0.15 \
    --flow_b_weight 2.0 \
    --seed 42 \
    --eval_seed 1042 \
    --expected_world_size "$NPROC" \
    --smoke_assertions \
    "$@"
