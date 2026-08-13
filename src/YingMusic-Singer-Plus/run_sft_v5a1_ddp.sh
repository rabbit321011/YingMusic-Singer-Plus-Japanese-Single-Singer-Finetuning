#!/bin/bash
# V5a1 Phase1 DDP 4卡启动脚本
# 论文复刻: text-only SFT (禁midi, 无CKA, 无text dropout)
# 从 V4c step_024000 出发, 24K步
# tmux session -d 'bash run_sft_v5a1_ddp.sh'
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_sft_v5a1_ddp.sh

set -euo pipefail

cd "$(dirname "$0")"

TOKEN_DIR="${SERVER_ROOT}/final_sum_large/pretreatment_text/tokens"
OUTPUT_DIR="ckpts/plus_ja_sft_v5a1"
BASE_CKPT="ckpts/plus_ja_sft_v4c/step_024000.pt"

echo "========================================"
echo "YingMusic-Plus V5a1 Phase1 SFT Training"
echo "Strategy: text-only (midi=zeros, no CKA, no text dropout)"
echo "Base:     $BASE_CKPT (V4c step_024000)"
echo "GPUs:     ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "Token dir: $TOKEN_DIR"
echo "Output:    $OUTPUT_DIR"
echo "LR: 1.4e-5 | warmup=500 | hold=12k | max_steps=24k"
echo "Loss: FlowA + 2×FlowB (no CKA)"
echo "========================================"

source "${CONDA_SH:-/path/to/conda.sh}"
conda activate yingmusic_plus

exec torchrun --nproc_per_node=4 train_plus_v5a1.py \
    --token_dir "$TOKEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --ckpt_path "$BASE_CKPT" \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 1.4e-5 \
    --warmup_steps 500 \
    --hold_steps 12000 \
    --max_steps 24000 \
    --max_duration 30 \
    --drop_text 0.0 \
    --cka_weight 0.0 \
    --flow_b_weight 2.0 \
    --seed 42 \
    "$@"

















