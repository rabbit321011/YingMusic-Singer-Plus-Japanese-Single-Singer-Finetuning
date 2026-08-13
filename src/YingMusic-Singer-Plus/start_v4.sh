#!/bin/bash
source "${CONDA_SH:-/path/to/conda.sh}"
conda activate yingmusic_plus
cd ${SERVER_ROOT}/YingMusic-Singer-Plus
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
echo '=== V4 Training Start ==='
echo "Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
python -c 'import torch; print(f"PyTorch {torch.__version__} CUDA {torch.cuda.is_available()}")'
echo '=== Launching ==='
exec bash run_sft_v4.sh 2>&1 | tee train_v4.log

















