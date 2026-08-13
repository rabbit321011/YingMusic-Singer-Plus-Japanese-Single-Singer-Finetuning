#!/bin/bash
# V4ca L1 fine-tune — from V4c step_024000
# 4卡 DDP | LR 8e-7 cosine | CKA=0.2 | 6k steps
set -euo pipefail
cd "$(dirname "$0")"

echo "========================================"
echo "YingMusic-Plus V4ca L1 Fine-tune"
echo "Resume: ckpts/plus_ja_sft_v4c/step_024000.pt"
echo "LR: 8e-7 cosine | CKA: 0.2 | 6k steps"
echo "Data: L1 only (7604 samples)"
echo "========================================"

exec torchrun --nproc_per_node=4 train_plus_v4ca.py \
    --token_dir ${SERVER_ROOT}/final_sum_large/pretreatment_text/tokens \
    --output_dir ckpts/plus_ja_sft_v4ca \
    --resume ckpts/plus_ja_sft_v4c/step_024000.pt \
    --batch_size 1 --grad_accum 4 --lr 8e-7 \
    --warmup_steps 100 --max_steps 6000 \
    --max_duration 30 --cka_weight 0.2 \
    --drop_text 0.15 --flow_b_weight 2.0 \
    --seed 42 "$@"

















