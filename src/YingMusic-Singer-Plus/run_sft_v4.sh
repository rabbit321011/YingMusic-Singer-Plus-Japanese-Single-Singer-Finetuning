#!/bin/bash
# V4 SFT 启动脚本
# 单卡模式（GPU set）: CUDA_VISIBLE_DEVICES=0 bash run_sft_v4.sh
# 多卡模式（4卡）:  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_sft_v4_ddp.sh

set -euo pipefail

cd "$(dirname "$0")"

TOKEN_DIR="${SERVER_ROOT}/final_sum_large/pretreatment_text/tokens"
OUTPUT_DIR="ckpts/plus_ja_sft_v4"

echo "========================================"
echo "YingMusic-Plus V4 SFT Training"
echo "Token dir: $TOKEN_DIR"
echo "Output:    $OUTPUT_DIR"
echo "========================================"

exec python train_plus_v4.py \
    --token_dir "$TOKEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 1 \
    --grad_accum 16 \
    --lr 7e-6 \
    --warmup_steps 500 \
    --max_steps 30000 \
    --max_duration 30 \
    --seed 42 \
    "$@"

















