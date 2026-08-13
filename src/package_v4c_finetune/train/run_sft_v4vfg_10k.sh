#!/bin/bash
# V4vfg: V4vf 30k DiT/EMA weights adapted to the 285k online VAE for 10k steps.

set -euo pipefail

cd "$(dirname "$0")"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate yingmusic_plus

SOURCE_CKPT="ckpts/plus_ja_sft_v4vf_lr1e4_30k/step_030000.pt"
BOOTSTRAP_CKPT="ckpts/v4vfg_bootstrap_from_v4vf_lr1e4_30k.pt"
VAE_CKPT="${REMOTE_ROOT}/experiments/vae_full_official_300k_20260716/checkpoints/autoencoder_285k.ckpt"
TOKEN_DIR="${REMOTE_ROOT}/final_sum_large/pretreatment_text/tokens_SOFA_v4d_control"
OUTPUT_DIR="ckpts/plus_ja_sft_v4vfg_v4vf30k_vae285k_10k"

for path in "$SOURCE_CKPT" "$VAE_CKPT"; do
    if [[ ! -f "$path" ]]; then
        echo "Required checkpoint is missing: $path" >&2
        exit 2
    fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
    exit 2
fi

if [[ ! -f "$BOOTSTRAP_CKPT" ]]; then
    python prepare_v4vfg_bootstrap.py \
        --source "$SOURCE_CKPT" \
        --output "$BOOTSTRAP_CKPT"
fi

echo "========================================"
echo "YingMusic-Plus V4vfg 10k"
echo "DiT/EMA source: $SOURCE_CKPT (experiment step reset to 0)"
echo "VAE:            $VAE_CKPT (285k online, frozen)"
echo "Output:         $OUTPUT_DIR"
echo "LR:             5e-6, warmup 250, hold 6000, cosine to 10k"
echo "GPUs:           ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "========================================"

export PYTHONUNBUFFERED=1
exec torchrun --nproc_per_node=4 train_plus_v4vf.py \
    --init_mode random \
    --resume "$BOOTSTRAP_CKPT" \
    --vae_ckpt "$VAE_CKPT" \
    --token_dir "$TOKEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 5e-6 \
    --warmup_steps 250 \
    --hold_steps 6000 \
    --max_steps 10000 \
    --max_duration 30 \
    --cka_weight 0.7 \
    --drop_text 0.15 \
    --flow_b_weight 2.0 \
    --seed 42 \
    "$@"
