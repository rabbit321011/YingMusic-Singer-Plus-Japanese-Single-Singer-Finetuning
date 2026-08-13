#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H_DATA_DIR="${V4IJPH_H_DATA_DIR:-${REMOTE_ROOT}/TEMP/h_v1_full_20260728_v2/training}"
CACHE_ROOT="${V4IJPH_INS_CACHE_ROOT:-${REMOTE_ROOT}/TEMP/v4ijph_ins_cache_v2}"
STYLE_ROOT="${V4IJPH_STYLE_ROOT:-${REMOTE_ROOT}/experiments/singing_style_embeddings}"
PYTHON="${V4IJPH_INS_PYTHON:-$STYLE_ROOT/envs/style_clap/bin/python}"
SOURCE_DIR="${V4IJPH_INS_SOURCE_DIR:-$STYLE_ROOT/models/paraspeechclap-src-4767eea9}"
SPEECH_MODEL_DIR="${V4IJPH_INS_SPEECH_MODEL_DIR:-${REMOTE_ROOT}/pretrained_models/wavlm-large}"
TEXT_MODEL_DIR="${V4IJPH_INS_TEXT_MODEL_DIR:-$STYLE_ROOT/models/granite-embedding-278m-multilingual-a9cb5338}"
CHECKPOINT_DIR="${V4IJPH_INS_CHECKPOINT_DIR:-$STYLE_ROOT/models/paraspeechclap-intrinsic-e80ad8eb}"
INVENTORY_DIR="${V4IJPH_INS_INVENTORY_DIR:-$STYLE_ROOT/outputs/audits/model_inventories}"
DEVICE="${V4IJPH_INS_DEVICE:-cuda:0}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

build_one() {
    local split="$1"
    local manifest="$2"
    local out_dir="$CACHE_ROOT/$split"
    echo "[V4IjPH cache v2] split=$split manifest=$manifest output=$out_dir device=$DEVICE"
    "$PYTHON" "$SCRIPT_DIR/build_v4ijph_ins_cache.py" \
        --manifest "$manifest" \
        --output-dir "$out_dir" \
        --source-dir "$SOURCE_DIR" \
        --source-inventory "$INVENTORY_DIR/paraspeechclap_source_inventory.json" \
        --speech-model-dir "$SPEECH_MODEL_DIR" \
        --speech-model-inventory "$INVENTORY_DIR/wavlm_large_local_model_inventory.json" \
        --text-model-dir "$TEXT_MODEL_DIR" \
        --text-model-inventory "$INVENTORY_DIR/granite_embedding_278m_model_inventory.json" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --checkpoint-inventory "$INVENTORY_DIR/paraspeechclap_intrinsic_model_inventory.json" \
        --device "$DEVICE" \
        --max-duration 30 \
        --flush-every 25 \
        --progress-every 10
}

audit_one() {
    local split="$1"
    local manifest="$2"
    shift 2
    "$PYTHON" "$SCRIPT_DIR/audit_v4ijph_ins_cache.py" \
        --cache-dir "$CACHE_ROOT/$split" \
        --manifest "$manifest" \
        --max-duration 30 \
        --progress-every 100 \
        "$@"
}

TRAIN_MANIFEST="$H_DATA_DIR/train_h_training.json"
EVAL_MANIFEST="$H_DATA_DIR/test_h_training.json"
build_one train "$TRAIN_MANIFEST"
build_one eval "$EVAL_MANIFEST"
audit_one train "$TRAIN_MANIFEST"
audit_one eval "$EVAL_MANIFEST" --skip-model-rehash
