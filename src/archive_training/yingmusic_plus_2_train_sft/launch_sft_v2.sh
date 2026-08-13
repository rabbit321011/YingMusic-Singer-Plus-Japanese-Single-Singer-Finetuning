#!/bin/bash
# YingMusic-Singer-Plus JA SFT v2 — CNENTokenizer (same as inference pipeline)
# 4 GPU (0,1,2,3), output to ckpts/plus_ja_sft_v2/
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PATH=${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin:$PATH
LOG_FILE="train_sft_v2_$(date +%Y%m%d_%H%M%S).log"
torchrun --nproc_per_node=4 train_plus_v2.py \
    --batch_size 6 --grad_accum 1 \
    --lr 7e-6 --warmup_steps 500 --max_steps 30000 \
    --log_every 50 --save_every 2000 --eval_every 1000 \
    --max_duration 15.0 --num_workers 0 --cka_weight 1.0 \
    --output_dir ckpts/plus_ja_sft_v2 \
    2>&1 | tee "$LOG_FILE"

