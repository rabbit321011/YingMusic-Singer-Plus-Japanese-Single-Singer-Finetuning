#!/bin/bash
# AR Ablation Experiments — 6 runs, sequential
# Effective batch 48/12 + LR 1e-4/5e-5/2.5e-5/6.25e-6
set -e

source ${CONDA_ROOT}
conda activate yysinger

SCRIPT=${REMOTE_ROOT}/train_ar_ablations.py
ACCELERATE=${REMOTE_ROOT}/.conda/envs/yysinger/bin/accelerate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_FILE=${REMOTE_ROOT}/ckpts/ar_ablations/run.log
echo "=== AR Ablation Experiments ===" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "[1/6] batch_48  eff=48  lr=2e-4  25ep" | tee -a "$LOG_FILE"
$ACCELERATE launch --num_processes=6 "$SCRIPT" \
    --run_name batch_48 --batch_size 4 --grad_accum 2 --lr 2e-4 --epochs 25 2>&1 | tee -a "$LOG_FILE"
echo "[1/6] batch_48  DONE at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "[2/6] batch_12  eff=12  lr=2e-4  25ep" | tee -a "$LOG_FILE"
$ACCELERATE launch --num_processes=6 "$SCRIPT" \
    --run_name batch_12 --batch_size 2 --grad_accum 1 --lr 2e-4 --epochs 25 2>&1 | tee -a "$LOG_FILE"
echo "[2/6] batch_12  DONE at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "[3/6] lr_1e-4   eff=96  lr=1e-4   50ep" | tee -a "$LOG_FILE"
$ACCELERATE launch --num_processes=6 "$SCRIPT" \
    --run_name lr_1e-4 --batch_size 4 --grad_accum 4 --lr 1e-4 --epochs 50 2>&1 | tee -a "$LOG_FILE"
echo "[3/6] lr_1e-4   DONE at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "[4/6] lr_5e-5   eff=96  lr=5e-5   50ep" | tee -a "$LOG_FILE"
$ACCELERATE launch --num_processes=6 "$SCRIPT" \
    --run_name lr_5e-5 --batch_size 4 --grad_accum 4 --lr 5e-5 --epochs 50 2>&1 | tee -a "$LOG_FILE"
echo "[4/6] lr_5e-5   DONE at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "[5/6] lr_2.5e-5 eff=96  lr=2.5e-5 50ep" | tee -a "$LOG_FILE"
$ACCELERATE launch --num_processes=6 "$SCRIPT" \
    --run_name lr_2.5e-5 --batch_size 4 --grad_accum 4 --lr 2.5e-5 --epochs 50 2>&1 | tee -a "$LOG_FILE"
echo "[5/6] lr_2.5e-5 DONE at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "[6/6] lr_6.25e-6 eff=96 lr=6.25e-6 50ep" | tee -a "$LOG_FILE"
$ACCELERATE launch --num_processes=6 "$SCRIPT" \
    --run_name lr_6.25e-6 --batch_size 4 --grad_accum 4 --lr 6.25e-6 --epochs 50 2>&1 | tee -a "$LOG_FILE"
echo "[6/6] lr_6.25e-6 DONE at $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "=== ALL 6 EXPERIMENTS DONE ===" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"

