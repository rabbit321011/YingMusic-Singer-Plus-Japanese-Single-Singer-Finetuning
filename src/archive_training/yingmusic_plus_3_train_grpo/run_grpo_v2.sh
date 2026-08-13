#!/bin/bash
# GRPO v2 训练
set -e
source ${CONDA_ROOT}
conda activate yingmusic_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ${REMOTE_ROOT}/YingMusic-Singer-Plus
rm -rf ${REMOTE_ROOT}/log_grpo_v2.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  scripts_archive/yingmusic_plus/3_train_grpo/train_grpo_v2.py \
    --token_dir ${REMOTE_ROOT}/final_sum_large/pretreatment_text/timeset \
    --output_dir ckpts/plus_grpo_v2 \
    --batch_size 1 \
    --lr 3e-6 \
    --max_steps 12000 \
    --save_every 100 \
    --log_every 1 \
    2>&1 | tee ${REMOTE_ROOT}/log_grpo_v2.txt

