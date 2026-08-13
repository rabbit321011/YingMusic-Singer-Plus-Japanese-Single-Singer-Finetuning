#!/bin/bash
# V4c SFT 启动脚本
# L1+L2 only | warmup-hold-decay | drop_text=15% | B区 weight=2.0
# 单卡: CUDA_VISIBLE_DEVICES=0 bash run_sft_v4c.sh

set -euo pipefail

cd "$(dirname "$0")"

TOKEN_DIR="${SERVER_ROOT}/final_sum_large/pretreatment_text/tokens"
OUTPUT_DIR="ckpts/plus_ja_sft_v4c"

echo "========================================"
echo "YingMusic-Plus V4c SFT Training"
echo "Data: L1+L2 only (no L3)"
echo "Token dir: $TOKEN_DIR"
echo "Output:    $OUTPUT_DIR"
echo "LR: 1.4e-5 | hold=12k | drop_text=15% | B_w=2.0 | CKA=0.7"
echo "========================================"

exec python train_plus_v4c.py \
    --token_dir "$TOKEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 1 \
    --grad_accum 16 \
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

















