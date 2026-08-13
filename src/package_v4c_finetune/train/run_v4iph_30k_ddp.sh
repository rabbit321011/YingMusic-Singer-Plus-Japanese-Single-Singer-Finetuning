#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${V4IPH_PROJECT_DIR:-${REMOTE_ROOT}/YingMusic-Singer-Plus}"
H_DATA_DIR="${V4IPH_H_DATA_DIR:-${REMOTE_ROOT}/TEMP/h_v1_full_20260728_v2/training}"
CACHE_DIR="${V4IPH_CACHE_DIR:-${REMOTE_ROOT}/TEMP/v4ph_game_cache_full_20260730_v2}"
TRAIN_SCRIPT="${V4IPH_TRAIN_SCRIPT:-$SCRIPT_DIR/train_v4iph.py}"
TORCHRUN="${V4IPH_TORCHRUN:-${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/torchrun}"
OUTPUT_DIR="${V4IPH_OUTPUT_DIR:-$PROJECT_DIR/ckpts/plus_ja_sft_v4iph}"
NPROC="${V4IPH_NPROC:-4}"

TRAIN_MANIFEST="$H_DATA_DIR/train_h_training.json"
EVAL_MANIFEST="$H_DATA_DIR/test_h_training.json"
TRAIN_SHA="b2560dc62eace7e294bce800e99a2669973fd3bceed42003c8033e7d8cb24467"
EVAL_SHA="befbf1341292f0c26707be6771178b381689c0de83682853e700bd9e1f5f38cc"
H_CONFIG="ca92066c0057e375e5d0485c54cb505ff8868a43e1b0efc528adcde6c1d11455"
PHASE_A_CHECKPOINT="$PROJECT_DIR/ckpts/v4ph_phase_a/step_000500_final.pt"
PHASE_A_SHA="d82eb26f31429c9ed52f5ad22c95162fa3f3dc4c60f15d9114cc2616bbd772ac"
ADJUDICATION="${REMOTE_ROOT}/TEMP/v4ph_phase_a_distance_adjudication_v2_20260730.json"
ADJUDICATION_SHA="890cf4db13f4ba16462618d82ccaecf1940394a71e7c4d5d246f742bfb60d74a"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================"
echo "V4IPH: V4PH with ref_len exactly zero"
echo "Condition: cond=0; full timeline uses H/PUL + GAME P"
echo "Loss: 2*FlowB(full) + 0.7*GAME-compatible CKA(full)"
echo "LR: 1.4e-5; warmup: 500; hold: 12000; steps: 30000"
echo "GPUs: $CUDA_VISIBLE_DEVICES (world=$NPROC)"
echo "Phase A: $PHASE_A_CHECKPOINT"
echo "Adjudication: $ADJUDICATION"
echo "Output: $OUTPUT_DIR"
echo "========================================"

cd "$PROJECT_DIR"

exec "$TORCHRUN" --nproc_per_node="$NPROC" "$TRAIN_SCRIPT" \
    --phase joint \
    --reference_mode all_b \
    --phase_a_checkpoint "$PHASE_A_CHECKPOINT" \
    --phase_a_checkpoint_sha256 "$PHASE_A_SHA" \
    --phase_a_adjudication "$ADJUDICATION" \
    --phase_a_adjudication_sha256 "$ADJUDICATION_SHA" \
    --train_manifest "$TRAIN_MANIFEST" \
    --eval_manifest "$EVAL_MANIFEST" \
    --train_manifest_sha256 "$TRAIN_SHA" \
    --eval_manifest_sha256 "$EVAL_SHA" \
    --h_config_fingerprint "$H_CONFIG" \
    --game_cache_manifest "$CACHE_DIR/manifest.json" \
    --placement_mode phone_pul \
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
    --num_workers 0 \
    --cka_weight 0.7 \
    --drop_text 0.15 \
    --flow_b_weight 2.0 \
    --seed 42 \
    --eval_seed 1042 \
    --expected_world_size "$NPROC" \
    --smoke_assertions \
    "$@"
