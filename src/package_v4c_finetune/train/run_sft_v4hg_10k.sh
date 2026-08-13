#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${H_PROJECT_DIR:-${REMOTE_ROOT}/YingMusic-Singer-Plus}"
DATA_DIR="${H_DATA_DIR:-${REMOTE_ROOT}/TEMP/h_v1_full_20260728_v2/training}"
TRAIN_SCRIPT="${H_TRAIN_SCRIPT:-$SCRIPT_DIR/train_plus_h.py}"
TORCHRUN="${H_TORCHRUN:-${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/torchrun}"
NPROC="${H_NPROC:-4}"

SOURCE_CKPT="${H_WARMSTART_CHECKPOINT:-$PROJECT_DIR/ckpts/plus_ja_sft_v4h/step_030000_final.pt}"
SOURCE_SHA="94339636ce3b4ac7416b4645474b78d300e375ec1fbd47f7d4a059ce8559387c"
VAE_CKPT="${H_VAE_CKPT:-${REMOTE_ROOT}/experiments/vae_full_official_300k_20260716/checkpoints/autoencoder_285k.ckpt}"
VAE_SHA="f18aeecacc04173cd2ea73bbdf8edae9e976d18e4ca050c38e2723281c5cba85"
OUTPUT_DIR="${H_OUTPUT_DIR:-$PROJECT_DIR/ckpts/plus_ja_sft_v4hg_10k}"

TRAIN_MANIFEST="$DATA_DIR/train_h_training.json"
EVAL_MANIFEST="$DATA_DIR/test_h_training.json"
TRAIN_SHA="b2560dc62eace7e294bce800e99a2669973fd3bceed42003c8033e7d8cb24467"
EVAL_SHA="befbf1341292f0c26707be6771178b381689c0de83682853e700bd9e1f5f38cc"
H_CONFIG="ca92066c0057e375e5d0485c54cb505ff8868a43e1b0efc528adcde6c1d11455"

for path in "$SOURCE_CKPT" "$VAE_CKPT" "$TRAIN_MANIFEST" "$EVAL_MANIFEST"; do
    if [[ ! -f "$path" ]]; then
        echo "Required input is missing: $path" >&2
        exit 2
    fi
done
if [[ -z "${H_RESUME:-}" && -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
    exit 2
fi
if [[ -n "${H_RESUME:-}" && ! -d "$OUTPUT_DIR" ]]; then
    echo "Resume output directory is missing: $OUTPUT_DIR" >&2
    exit 2
fi
ACTUAL_VAE_SHA="$(sha256sum "$VAE_CKPT" | awk '{print $1}')"
if [[ "$ACTUAL_VAE_SHA" != "$VAE_SHA" ]]; then
    echo "VAE SHA256 mismatch: $ACTUAL_VAE_SHA != $VAE_SHA" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

STOP_ARGS=()
if [[ -n "${H_STOP_AFTER_STEP:-}" ]]; then
    STOP_ARGS+=(--stop_after_step "$H_STOP_AFTER_STEP")
fi
RESUME_ARGS=()
if [[ -n "${H_RESUME:-}" ]]; then
    RESUME_ARGS+=(--resume "$H_RESUME")
fi
EXTRA_ARGS=()
if [[ "${H_SMOKE_ASSERTIONS:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--smoke_assertions)
fi
if [[ "${H_OVERFIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--overfit --overfit_n "${H_OVERFIT_N:-20}")
fi

echo "========================================"
echo "V4Hg 285k-VAE adaptation"
echo "DiT/EMA source: $SOURCE_CKPT"
echo "VAE:            $VAE_CKPT"
echo "GPUs:           $CUDA_VISIBLE_DEVICES (world=$NPROC)"
echo "Output:         $OUTPUT_DIR"
echo "Steps:          ${H_MAX_STEPS:-10000}; stop: ${H_STOP_AFTER_STEP:-none}"
echo "LR:             5e-6; warmup 250; hold 6000"
echo "========================================"

cd "$PROJECT_DIR"

exec "$TORCHRUN" --nproc_per_node="$NPROC" "$TRAIN_SCRIPT" \
    --train_manifest "$TRAIN_MANIFEST" \
    --eval_manifest "$EVAL_MANIFEST" \
    --train_manifest_sha256 "$TRAIN_SHA" \
    --eval_manifest_sha256 "$EVAL_SHA" \
    --h_config_fingerprint "$H_CONFIG" \
    --placement_mode phone_pul \
    --output_dir "$OUTPUT_DIR" \
    --ckpt_path ckpts/YingMusicSinger_model.pt \
    --warmstart_checkpoint "$SOURCE_CKPT" \
    --warmstart_expected_sha256 "$SOURCE_SHA" \
    --warmstart_expected_step 30000 \
    --vae_ckpt "$VAE_CKPT" \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 5e-6 \
    --warmup_steps 250 \
    --hold_steps 6000 \
    --max_steps "${H_MAX_STEPS:-10000}" \
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
    "${RESUME_ARGS[@]}" \
    "$@"
