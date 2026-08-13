#!/bin/bash
# V4b SFT 启动脚本
# 从官方 base 权重从头训练，新参数：lr=1.4e-5, cka=0.7
# 单卡: CUDA_VISIBLE_DEVICES=0 bash run_sft_v4b.sh

set -euo pipefail

cd "$(dirname "$0")"

TOKEN_DIR="${SERVER_ROOT}/final_sum_large/pretreatment_text/tokens"
OUTPUT_DIR="ckpts/plus_ja_sft_v4b"

echo "========================================"
echo "YingMusic-Plus V4b SFT Training (from scratch)"
echo "Token dir:   $TOKEN_DIR"
echo "Output:      $OUTPUT_DIR"
echo "LR: 1.4e-5 | CKA: 0.7"
echo "========================================"

exec python train_plus_v4b.py \
    --token_dir "$TOKEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 1 \
    --grad_accum 16 \
    --lr 1.4e-5 \
    --warmup_steps 500 \
    --max_steps 30000 \
    --max_duration 30 \
    --cka_weight 0.7 \
    --seed 42 \
    "$@"

















