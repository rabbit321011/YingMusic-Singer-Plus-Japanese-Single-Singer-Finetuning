#!/bin/bash
# V4d P0.5: V4c-24k weights + SOFA timestamps, fresh optimizer and LR schedule.
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_sft_v4d_ddp.sh

set -euo pipefail

cd "$(dirname "$0")"

TOKEN_DIR="${REMOTE_ROOT}/final_sum_large/pretreatment_text/tokens_SOFA_v4d_control"
OUTPUT_DIR="ckpts/plus_ja_sft_v4d_sofa"
INIT_CKPT="ckpts/plus_ja_sft_v4c/step_024000.pt"

echo "========================================"
echo "YingMusic-Plus V4d P0.5 SOFA falsification run"
echo "Initialize from: $INIT_CKPT"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "Token dir:  $TOKEN_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Fresh LR: warmup 500 -> hold 1.4e-5 | drop_text=15% | B_w=2.0 | CKA=0.7"
echo "========================================"

exec torchrun --nproc_per_node=4 train_plus_v4d.py \
    --token_dir "$TOKEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --init_checkpoint "$INIT_CKPT" \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 1.4e-5 \
    --warmup_steps 500 \
    --hold_steps 10000 \
    --max_steps 10000 \
    --max_duration 30 \
    --cka_weight 0.7 \
    --drop_text 0.15 \
    --flow_b_weight 2.0 \
    --seed 42 \
    "$@"
