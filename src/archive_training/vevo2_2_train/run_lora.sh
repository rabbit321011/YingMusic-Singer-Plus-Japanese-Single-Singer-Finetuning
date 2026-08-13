#!/bin/bash
source ${CONDA_ROOT}
conda activate yysinger
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p ${REMOTE_ROOT}/ckpts/ar_lora
${REMOTE_ROOT}/.conda/envs/yysinger/bin/accelerate launch --num_processes=6 ${REMOTE_ROOT}/train_ar_lora.py 2>&1 | tee ${REMOTE_ROOT}/ckpts/ar_lora/run.log

