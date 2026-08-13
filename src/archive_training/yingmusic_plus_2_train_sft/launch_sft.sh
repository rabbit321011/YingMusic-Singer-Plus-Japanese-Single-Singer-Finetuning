#!/bin/bash
# YingMusic-Singer-Plus SFT Training (4GPU)
LOG_FILE="train_sft_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 train_plus.py \
    --batch_size 6 --grad_accum 1 \
    --lr 7e-6 --warmup_steps 500 --max_steps 30000 \
    --log_every 50 --save_every 2000 --eval_every 1000 \
    --max_duration 15.0 --num_workers 0 --cka_weight 1.0 \
    --output_dir ckpts/plus_ja_sft \
    2>&1 | tee "$LOG_FILE"

