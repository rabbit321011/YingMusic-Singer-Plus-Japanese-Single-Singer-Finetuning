#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "sentence" && "$MODE" != "phone" && "$MODE" != "phone_pul" ]]; then
    echo "usage: $0 {sentence|phone|phone_pul} [train_plus_h.py arguments...]" >&2
    exit 2
fi
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${H_PROJECT_DIR:-${REMOTE_ROOT}/YingMusic-Singer-Plus}"
DATA_DIR="${H_DATA_DIR:-${REMOTE_ROOT}/TEMP/h_v1_full_20260728_v2/training}"
TRAIN_SCRIPT="${H_TRAIN_SCRIPT:-$SCRIPT_DIR/train_plus_h.py}"
TORCHRUN="${H_TORCHRUN:-${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/torchrun}"
NPROC="${H_NPROC:-4}"

TRAIN_MANIFEST="$DATA_DIR/train_h_training.json"
EVAL_MANIFEST="$DATA_DIR/test_h_training.json"
TRAIN_SHA="b2560dc62eace7e294bce800e99a2669973fd3bceed42003c8033e7d8cb24467"
EVAL_SHA="befbf1341292f0c26707be6771178b381689c0de83682853e700bd9e1f5f38cc"
H_CONFIG="ca92066c0057e375e5d0485c54cb505ff8868a43e1b0efc528adcde6c1d11455"
OUTPUT_DIR="${H_OUTPUT_DIR:-$PROJECT_DIR/ckpts/h_v1_${MODE}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

EXTRA_ARGS=()
if [[ "${H_SMOKE_ASSERTIONS:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--smoke_assertions)
fi
if [[ "${H_OVERFIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--overfit --overfit_n "${H_OVERFIT_N:-20}")
fi

echo "========================================"
echo "H/V4H corrected-DDP training"
echo "Mode: $MODE"
echo "GPUs: $CUDA_VISIBLE_DEVICES (world=$NPROC)"
echo "Manifest: $TRAIN_MANIFEST"
echo "Output: $OUTPUT_DIR"
echo "Steps: ${H_MAX_STEPS:-30000}; stop: ${H_STOP_AFTER_STEP:-none}"
echo "========================================"

cd "$PROJECT_DIR"

STOP_ARGS=()
if [[ -n "${H_STOP_AFTER_STEP:-}" ]]; then
    STOP_ARGS+=(--stop_after_step "$H_STOP_AFTER_STEP")
fi

exec "$TORCHRUN" --nproc_per_node="$NPROC" "$TRAIN_SCRIPT" \
    --train_manifest "$TRAIN_MANIFEST" \
    --eval_manifest "$EVAL_MANIFEST" \
    --train_manifest_sha256 "$TRAIN_SHA" \
    --eval_manifest_sha256 "$EVAL_SHA" \
    --h_config_fingerprint "$H_CONFIG" \
    --placement_mode "$MODE" \
    --output_dir "$OUTPUT_DIR" \
    --ckpt_path ckpts/YingMusicSinger_model.pt \
    --batch_size 1 \
    --grad_accum "${H_GRAD_ACCUM:-4}" \
    --lr 1.4e-5 \
    --warmup_steps 500 \
    --hold_steps 12000 \
    --max_steps "${H_MAX_STEPS:-30000}" \
    --save_every "${H_SAVE_EVERY:-2000}" \
    --eval_every "${H_EVAL_EVERY:-1000}" \
    --log_every "${H_LOG_EVERY:-50}" \
    --max_duration 30 \
    --num_workers 0 \
    --cka_weight 0.7 \
    --drop_text 0.15 \
    --flow_b_weight 2.0 \
    --seed 42 \
    --eval_seed 1042 \
    --expected_world_size "$NPROC" \
    "${EXTRA_ARGS[@]}" \
    "${STOP_ARGS[@]}" \
    "$@"
