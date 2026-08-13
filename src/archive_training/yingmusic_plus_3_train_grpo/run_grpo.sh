#!/bin/bash
# ============================================================
# GRPO 训练启动脚本 (v2: Reward Server 分离部署)
#
# GPU 分配:
#   GPU set: DDP 训练 (DiT + VAE + SOME)
#   GPU set:   Reward Server (Whisper + WavLM + torchcrepe + DNSMOS)
#
# 用法: bash run_grpo.sh [--tmux session
# ============================================================

set -euo pipefail

# ---- 环境 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${REMOTE_ROOT}/YingMusic-Singer-Plus"
CONDA_PATH="${CONDA_ROOT}"
CONDA_ENV="yingmusic_plus"

# 服务器路径
CKPT_PATH="${PROJECT_DIR}/ckpts/plus_ja_sft_v4c/step_024000.pt"
VAE_CONFIG="${PROJECT_DIR}/src/YingMusicSinger/config/stable_audio_2_0_vae_20hz_official.json"
VAE_CKPT="${PROJECT_DIR}/ckpts/stable_audio_2_0_vae_20hz_official.ckpt"
MIDI_CKPT="${PROJECT_DIR}/ckpts/model_ckpt_steps_100000_simplified.ckpt"
CONFIG="${PROJECT_DIR}/src/YingMusicSinger/config/YingMusic_Singer.yaml"
TOKEN_DIR="${REMOTE_ROOT}/final_sum_large/pretreatment_text/timeset"
OUTPUT_DIR="${PROJECT_DIR}/ckpts/plus_grpo_v1"
TMP_DIR="/tmp/grpo_wavs"
REWARD_PORT=15555
REWARD_GPU=4  # 专用 reward GPU

# 训练超参数
TRAIN_GPUS=4
BATCH_SIZE=6       # per GPU
MAX_STEPS=4800
LR=7e-6
SAVE_EVERY=500
LOG_EVERY=10
SEED=42

# ---- 显式 PATH ----
export PATH="/usr/bin:$HOME/bin:$HOME/.local/bin:$PATH"
export HF_ENDPOINT="https://hf-mirror.com"

# ---- Conda ----
if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
    conda activate "$CONDA_ENV"
else
    echo "WARNING: conda not found at $CONDA_PATH"
fi

# ---- 目录 ----
cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR" "$TMP_DIR"

# ---- 命令 ----

# Reward Server (GPU set)
SERVER_CMD="CUDA_VISIBLE_DEVICES=${REWARD_GPU} python ${SCRIPT_DIR}/reward_server.py --port ${REWARD_PORT}"

# 训练 (GPU set)
TRAIN_CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=${TRAIN_GPUS} \
    --master_port=$((29500 + RANDOM % 1000)) \
    ${SCRIPT_DIR}/train_grpo.py \
    --config \"${CONFIG}\" \
    --ckpt_path \"${CKPT_PATH}\" \
    --vae_config \"${VAE_CONFIG}\" \
    --vae_ckpt \"${VAE_CKPT}\" \
    --midi_ckpt \"${MIDI_CKPT}\" \
    --token_dir \"${TOKEN_DIR}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --max_steps ${MAX_STEPS} \
    --save_every ${SAVE_EVERY} \
    --log_every ${LOG_EVERY} \
    --seed ${SEED} \
    --tmp_dir \"${TMP_DIR}\" \
    --save_baseline_1200"

# ---- 打印 ----
echo "============================================================"
echo " GRPO Training v2 (Reward Server on GPU ${REWARD_GPU})"
echo "============================================================"
echo " Project:    ${PROJECT_DIR}"
echo " Train GPU:  0-3 (DDP x${TRAIN_GPUS})"
echo " Reward GPU: ${REWARD_GPU} (port ${REWARD_PORT})"
echo " Batch/GPU:  ${BATCH_SIZE}"
echo " Steps:      ${MAX_STEPS} (save baseline @1200)"
echo " Output:     ${OUTPUT_DIR}"
echo "============================================================"

# ---- 启动 ----
if [[ "${1:-}" == "--tmux session ]]; then
    SESSION="grpo_train"
    tmux session -t "$SESSION" 2>/dev/null && tmux session -t "$SESSION"

    echo "Launching in tmux session
    echo "  Attach:  tmux session -t ${SESSION}"
    echo "  Detach:  Ctrl+B, D"
    echo ""

    tmux session -d -s "$SESSION" -n server "bash -c '
echo \"=== Reward Server starting on GPU ${REWARD_GPU}... ===\"
${SERVER_CMD} 2>&1 | tee ${OUTPUT_DIR}/reward_server_\$(date +%Y%m%d_%H%M%S).log
exec bash
'"

    # 等 server 就绪（检查最新 log + TCP 端口）
    echo "Waiting for reward server to be ready..."
    for i in $(seq 1 180); do
        # 检查最新 log 中是否有 "Listening on"
        if grep -q "Listening on" "${OUTPUT_DIR}"/reward_server_*.log 2>/dev/null; then
            echo "Reward server ready after ${i}s"
            break
        fi
        # 也可以直接检查端口
        if ss -tlnp 2>/dev/null | grep -q ":${REWARD_PORT}"; then
            echo "Reward server ready (port check) after ${i}s"
            break
        fi
        sleep 1
    done

    # 创建训练窗口
    tmux session -h -t "$SESSION" "bash -c '
echo \"=== GRPO Training starting (GPU set)... ===\"
echo \"\"
eval ${TRAIN_CMD} 2>&1 | tee ${OUTPUT_DIR}/train_grpo_\$(date +%Y%m%d_%H%M%S).log
echo \"\"
echo \"=== GRPO Training Finished ===\"
# 训练结束后杀掉 server
tmux session -t ${SESSION} 2>/dev/null
exec bash
'"

    echo "Tmux session '$SESSION' started."
    echo "  Left pane:  Reward Server (GPU ${REWARD_GPU})"
    echo "  Right pane: Training (GPU set)"
    echo "  Use 'tmux session -t $SESSION' to view."

else
    # 直接运行（无 tmux session
    echo "Starting reward server on GPU ${REWARD_GPU}..."
    eval "$SERVER_CMD" &
    SERVER_PID=$!
    trap "kill $SERVER_PID 2>/dev/null" EXIT

    echo "Waiting for server..."
    sleep 15

    echo "Starting training on GPU set..."
    eval "$TRAIN_CMD" 2>&1 | tee "${OUTPUT_DIR}/train_grpo_$(date +%Y%m%d_%H%M%S).log"

    kill $SERVER_PID 2>/dev/null
    echo "Done."
fi

