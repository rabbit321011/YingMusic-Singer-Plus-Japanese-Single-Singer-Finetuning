#!/bin/bash
# V4vf: random DiT initialization + V4f SOFA/fuzz-MIDI/official-VAE recipe.
# Example smoke:
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_sft_v4vf_random_init_ddp.sh \
#   --output_dir ckpts/plus_ja_sft_v4vf_smoke --max_steps 10 \
#   --grad_accum 1 --log_every 1 --save_every 10 --eval_every 1000

set -euo pipefail

cd "$(dirname "$0")"

TOKEN_DIR="${REMOTE_ROOT}/final_sum_large/pretreatment_text/tokens_SOFA_v4d_control"
OUTPUT_DIR="ckpts/plus_ja_sft_v4vf"

echo "========================================"
echo "YingMusic-Plus V4vf random-init run"
echo "Init mode: random (no Official/V4 checkpoint)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "Token dir:  $TOKEN_DIR"
echo "Output:     $OUTPUT_DIR"
echo "MIDI: continuous SOME latent + MIDIFuzzDisturb"
echo "VAE: official frozen VAE"
echo "LR: warmup 500 -> hold 12k -> cosine decay at 30k"
echo "========================================"

exec torchrun --nproc_per_node=4 train_plus_v4vf.py \
    --init_mode random \
    --token_dir "$TOKEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 1.4e-5 \
    --warmup_steps 500 \
    --hold_steps 12000 \
    --max_steps 30000 \
    --max_duration 30 \
    --cka_weight 0.7 \
    --drop_text 0.15 \
    --flow_b_weight 2.0 \
    --seed 42 \
    "$@"
