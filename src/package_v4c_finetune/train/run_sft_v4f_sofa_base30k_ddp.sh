#!/bin/bash
# V4f: official base model + SOFA timestamps, 30k V4c recipe.
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_sft_v4f_sofa_base30k_ddp.sh

set -euo pipefail

cd "$(dirname "$0")"

TOKEN_DIR="${REMOTE_ROOT}/final_sum_large/pretreatment_text/tokens_SOFA_v4d_control"
OUTPUT_DIR="ckpts/plus_ja_sft_v4f_sofa_base30k"
BASE_CKPT="ckpts/YingMusicSinger_model.pt"

echo "========================================"
echo "YingMusic-Plus V4f SOFA base-model 30k run"
echo "Base checkpoint: $BASE_CKPT"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "Token dir:  $TOKEN_DIR"
echo "Output:     $OUTPUT_DIR"
echo "LR: warmup 500 -> hold 12k -> cosine decay at 30k"
echo "========================================"

exec torchrun --nproc_per_node=4 train_plus_v4d.py \
    --token_dir "$TOKEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --ckpt_path "$BASE_CKPT" \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 1.4e-5 \
    --warmup_steps 500 \
    --hold_steps 12000 \
    --max_steps 30000 \
    --max_duration 30 \
    --cka_weight 0.7 \
    --drop_text 0.15 \
    --flow_b_weight 2.0 \
    --seed 42 \
    "$@"
