#!/bin/bash
# V4c DDP 4卡 resume 启动脚本
# 从 step 4000 checkpoint 继续，LR 曲线自然衔接
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_sft_v4c_ddp.sh

set -euo pipefail

cd "$(dirname "$0")"

TOKEN_DIR="${REMOTE_ROOT}/final_sum_large/pretreatment_text/tokens"
OUTPUT_DIR="ckpts/plus_ja_sft_v4c"
RESUME_CKPT="ckpts/plus_ja_sft_v4c/step_004000.pt"

echo "========================================"
echo "YingMusic-Plus V4c DDP Training (resume)"
echo "Resume from: $RESUME_CKPT"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "Token dir:  $TOKEN_DIR"
echo "Output:     $OUTPUT_DIR"
echo "LR: 1.4e-5 | hold=12k | drop_text=15% | B_w=2.0 | CKA=0.7"
echo "========================================"

exec torchrun --nproc_per_node=4 train_plus_v4c.py \
    --token_dir "$TOKEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resume "$RESUME_CKPT" \
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
