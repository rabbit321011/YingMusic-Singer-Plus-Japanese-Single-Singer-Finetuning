#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${V5SG_PROJECT_DIR:-${REMOTE_ROOT}/YingMusic-Singer-Plus}"
FROZEN_DIR="${V5SG_FROZEN_DIR:-${REMOTE_ROOT}/V5P_20260808/data/frozen}"
TRAIN_SCRIPT="${V5SG_TRAIN_SCRIPT:-$SCRIPT_DIR/train_v5sg.py}"
TORCHRUN="${V5SG_TORCHRUN:-${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/torchrun}"
NPROC="${V5SG_NPROC:-4}"

OFFICIAL_CKPT="${V5SG_OFFICIAL_CKPT:-$PROJECT_DIR/ckpts/YingMusicSinger_model.pt}"
OFFICIAL_SHA="8b0684557abcdb00ee9e39cb351d5b9b90b98773ab6630d00fa27daa0e51ab56"
VAE_CKPT="${V5SG_VAE_CKPT:-${REMOTE_ROOT}/experiments/vae_full_official_300k_20260716/checkpoints/autoencoder_285k.ckpt}"
VAE_SHA="f18aeecacc04173cd2ea73bbdf8edae9e976d18e4ca050c38e2723281c5cba85"
MIDI_CKPT="${V5SG_MIDI_CKPT:-$PROJECT_DIR/ckpts/model_ckpt_steps_100000_simplified.ckpt}"
MIDI_SHA="aa710fce920b4dae281b0e6cc2acba83345d82ee62d51f7bafeb29636f28f97c"
OUTPUT_DIR="${V5SG_OUTPUT_DIR:-$PROJECT_DIR/ckpts/plus_ja_sft_v5sg}"

TRAIN_MANIFEST="$FROZEN_DIR/train_mixed.json"
EVAL_MANIFEST="$FROZEN_DIR/eval_mixed.json"
POOL_AUDIT="$FROZEN_DIR/pool_audit.json"
TRAIN_SHA="f4ed5804251691348b19c88e2aa26c5fcc9dcd820ccc032aecd220132bb2ee98"
EVAL_SHA="2b926b08ea8f8ca0f13f8c0e5dfe0d13186026ad0a91035edbb4d5105dae8bc1"
SHORT_SHA="f75b84f86701fa6d804f090bbb07e595312fe0d39f09eb409568d389c599f2b4"
LONG_SHA="599184a7c513812e574414dee5ecc28ead23c231c0b4411ef9caebc51fe0c24a"
POOL_AUDIT_SHA="3a8f702bfc0c82b67378e371293dd5d3c0c832f783c9d17647eb5c2ddf89812e"
H_CONFIG="67eae32578eeb9a8d0c338dfb6e724c697ad8c3578ec061c0ec3a0f52974dd07"

for path in "$OFFICIAL_CKPT" "$VAE_CKPT" "$MIDI_CKPT" \
            "$TRAIN_MANIFEST" "$EVAL_MANIFEST" "$POOL_AUDIT" "$TRAIN_SCRIPT"; do
    if [[ ! -f "$path" ]]; then
        echo "Required input is missing: $path" >&2
        exit 2
    fi
done
if [[ -z "${V5SG_RESUME:-}" && -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
    exit 2
fi
if [[ -n "${V5SG_RESUME:-}" && ! -d "$OUTPUT_DIR" ]]; then
    echo "Resume output directory is missing: $OUTPUT_DIR" >&2
    exit 2
fi

for spec in "$OFFICIAL_CKPT:$OFFICIAL_SHA" "$VAE_CKPT:$VAE_SHA" \
            "$MIDI_CKPT:$MIDI_SHA" "$TRAIN_MANIFEST:$TRAIN_SHA" \
            "$EVAL_MANIFEST:$EVAL_SHA" "$POOL_AUDIT:$POOL_AUDIT_SHA"; do
    path="${spec%:*}"
    expected="${spec##*:}"
    actual="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
        echo "SHA256 mismatch: $path: $actual != $expected" >&2
        exit 2
    fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

STOP_ARGS=()
if [[ -n "${V5SG_STOP_AFTER_STEP:-}" ]]; then
    STOP_ARGS+=(--stop_after_step "$V5SG_STOP_AFTER_STEP")
fi
RESUME_ARGS=()
if [[ -n "${V5SG_RESUME:-}" ]]; then
    RESUME_ARGS+=(--resume "$V5SG_RESUME")
fi
EXTRA_ARGS=()
if [[ "${V5SG_SMOKE_ASSERTIONS:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--smoke_assertions)
fi
if [[ "${V5SG_OVERFIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--overfit --overfit_n "${V5SG_OVERFIT_N:-20}")
fi

echo "========================================"
echo "V5-Sg official-fresh 285k + continuous SOME"
echo "Official source: $OFFICIAL_CKPT"
echo "VAE:             $VAE_CKPT"
echo "SOME:            $MIDI_CKPT"
echo "GPUs:            $CUDA_VISIBLE_DEVICES (world=$NPROC)"
echo "Output:          $OUTPUT_DIR"
echo "Steps:           40000; stop: ${V5SG_STOP_AFTER_STEP:-none}"
echo "LR:              0--4k 0->1.4e-5; 4k--28k ->1e-5; 28k--40k ->0"
echo "========================================"

cd "$PROJECT_DIR"

exec "$TORCHRUN" --nproc_per_node="$NPROC" "$TRAIN_SCRIPT" \
    --train_manifest "$TRAIN_MANIFEST" \
    --eval_manifest "$EVAL_MANIFEST" \
    --train_manifest_sha256 "$TRAIN_SHA" \
    --eval_manifest_sha256 "$EVAL_SHA" \
    --short_manifest_sha256 "$SHORT_SHA" \
    --long_manifest_sha256 "$LONG_SHA" \
    --pool_audit "$POOL_AUDIT" \
    --pool_audit_sha256 "$POOL_AUDIT_SHA" \
    --h_config_fingerprint "$H_CONFIG" \
    --placement_mode phone_pul \
    --output_dir "$OUTPUT_DIR" \
    --ckpt_path "$OFFICIAL_CKPT" \
    --vae_ckpt "$VAE_CKPT" \
    --midi_ckpt "$MIDI_CKPT" \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 1.4e-5 \
    --warmup_steps 4000 \
    --hold_steps 0 \
    --first_decay_end 28000 \
    --mid_lr 1e-5 \
    --max_steps 40000 \
    --save_every "${V5SG_SAVE_EVERY:-2000}" \
    --eval_every "${V5SG_EVAL_EVERY:-1000}" \
    --log_every "${V5SG_LOG_EVERY:-25}" \
    --max_duration 60.1 \
    --num_workers 0 \
    --cka_weight 0.7 \
    --drop_text 0.15 \
    --flow_b_weight 2.0 \
    --seed 42 \
    --eval_seed 1042 \
    --expected_world_size "$NPROC" \
    --ema_device cpu \
    "${EXTRA_ARGS[@]}" \
    "${STOP_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "$@"
