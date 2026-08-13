#!/bin/bash
# Launch DiT fine-tune in tmux session

cd ${REMOTE_ROOT}/YingMusic-Singer

source ${CONDA_ROOT}
conda activate yysinger

export LD_LIBRARY_PATH=${REMOTE_ROOT}/.conda/envs/yysinger/lib
export ESPEAK_DATA_PATH=${REMOTE_ROOT}/.conda/envs/yysinger/share/espeak-ng-data
export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2,3,4,5
torchrun --nproc_per_node=4 train_v2.py \
    --train_json ${REMOTE_ROOT}/final_sum_large/train_singnet.json \
    --test_json  ${REMOTE_ROOT}/final_sum_large/test_singnet.json \
    --output_dir ${REMOTE_ROOT}/YingMusic-Singer/finetune_v2_lr2e5 \
    --singer_path ${REMOTE_ROOT}/YingMusic-Singer \
    --language ja \
    --lr 2e-5 \
    --lr_min 1e-6 \
    --warmup_steps 3000 \
    --max_steps 50000 \
    --save_every 2000 \
    --eval_every 2000 \
    --seed 42 \
    2>&1 | tee finetune_v2.log

echo "Done. Check finetune_v2.log"

