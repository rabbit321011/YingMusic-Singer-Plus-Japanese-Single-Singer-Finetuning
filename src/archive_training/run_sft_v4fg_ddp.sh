#!/bin/bash
# V4fg: V4f 24k DiT + 285k VAE, 15k steps.
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_sft_v4fg_ddp.sh

set -euo pipefail

cd "$(dirname "$0")"

TOKEN_DIR="${REMOTE_ROOT}/final_sum_large/pretreatment_text/tokens_SOFA_v4d_control"
OUTPUT_DIR="ckpts/plus_ja_sft_v4fg"
BASE_CKPT="ckpts/plus_ja_sft_v4f_sofa_base30k/step_024000.pt"

echo "========================================"
echo "YingMusic-Plus V4fg: V4f 24k + 285k VAE 15k run"
echo "Base checkpoint: $BASE_CKPT"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "Token dir:  $TOKEN_DIR"
echo "Output:     $OUTPUT_DIR"
echo "LR: warmup 250 -> hold 6k -> cosine decay at 15k"
echo "========================================"

exec torchrun --nproc_per_node=4 train_plus_v4g.py \
    --token_dir "$TOKEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --ckpt_path "$BASE_CKPT" \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 5e-6 \
    --warmup_steps 250 \
    --hold_steps 6000 \
    --max_steps 15000 \
    --max_duration 30 \
    --cka_weight 0.7 \
    --drop_text 0.15 \
    --flow_b_weight 2.0 \
    --seed 42 \
    "$@"

