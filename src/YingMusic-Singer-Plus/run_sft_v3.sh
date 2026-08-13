#!/bin/bash
cd ${SERVER_ROOT}/YingMusic-Singer-Plus
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_plus_v3.py \
  --train_json ${SERVER_ROOT}/final_sum_large/train_singnet.json \
  --eval_json ${SERVER_ROOT}/final_sum_large/test_singnet.json \
  --output_dir ckpts/plus_ja_sft_v3 \
  --batch_size 4 --lr 7e-6 --warmup_steps 500 --max_steps 30000 \
  --max_duration 30 --seed 42 \
  "$@"

















