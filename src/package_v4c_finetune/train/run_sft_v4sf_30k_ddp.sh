#!/bin/bash
# V4Sf formal: V4f recipe, SVC-self-cloned train audio, corrected gradient sync.

set -euo pipefail

cd "$(dirname "$0")"

TRAIN_TOKEN_DIR="${REMOTE_ROOT}/final_sum_large/pretreatment_text/tokens_SOFA_v4sf"
ORIGINAL_EVAL_TOKEN_DIR="${REMOTE_ROOT}/final_sum_large/pretreatment_text/tokens_SOFA_v4d_control"
SVC_EVAL_TOKEN_DIR="${REMOTE_ROOT}/final_sum_large/pretreatment_text/tokens_SOFA_v4sf_eval"
OUTPUT_DIR="ckpts/plus_ja_sft_v4sf_sofa_base30k"

echo "========================================"
echo "YingMusic-Plus V4Sf corrected-DDP formal 30k"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "Train tokens: $TRAIN_TOKEN_DIR"
echo "Eval original: $ORIGINAL_EVAL_TOKEN_DIR"
echo "Eval SVC: $SVC_EVAL_TOKEN_DIR"
echo "Output: $OUTPUT_DIR"
echo "LR: warmup 500 -> hold 12k -> cosine decay at 30k"
echo "========================================"

exec ${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/torchrun --nproc_per_node=4 train_plus_v4sf.py \
    --token_dir "$TRAIN_TOKEN_DIR" \
    --eval_token_dir "$ORIGINAL_EVAL_TOKEN_DIR" \
    --eval_svc_token_dir "$SVC_EVAL_TOKEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --ckpt_path ckpts/YingMusicSinger_model.pt \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 1.4e-5 \
    --warmup_steps 500 \
    --hold_steps 12000 \
    --max_steps 30000 \
    --save_every 2000 \
    --eval_every 1000 \
    --log_every 50 \
    --max_duration 30 \
    --cka_weight 0.7 \
    --drop_text 0.15 \
    --flow_b_weight 2.0 \
    --seed 42 \
    --eval_seed 1042 \
    "$@"
