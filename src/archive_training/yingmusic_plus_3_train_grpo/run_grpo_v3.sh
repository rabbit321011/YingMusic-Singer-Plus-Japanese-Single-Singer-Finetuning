#!/bin/bash
# GRPO v3 训练 — Whisper 外包给 GPU set 训练跑 GPU set
set -e
source ${CONDA_ROOT}
conda activate yingmusic_plus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ${REMOTE_ROOT}/YingMusic-Singer-Plus
rm -f ${REMOTE_ROOT}/log_grpo_v3.txt

# ---- 启动 Whisper Server (GPU set) ----
echo "[Launcher] Starting Whisper server on GPU set..."
CUDA_VISIBLE_DEVICES=3 python scripts_archive/yingmusic_plus/3_train_grpo/reward_server.py \
    --port 15555 &
SERVER_PID=$!
echo "[Launcher] Server PID=$SERVER_PID, waiting for ready (may take 30-60s)..."
sleep 30
# Poll until server is ready
for i in $(seq 1 30); do
    if python -c "import socket; s=socket.socket(); s.settimeout(2); r=s.connect_ex(('127.0.0.1',15555)); s.close(); exit(r)" 2>/dev/null; then
        echo "[Launcher] Server ready after $((30+i))s."
        break
    fi
    sleep 2
done
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[Launcher] ERROR: Server failed!"
    exit 1
fi
echo "[Launcher] Server ready."

# ---- 训练 (GPU set) ----
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 \
  scripts_archive/yingmusic_plus/3_train_grpo/train_grpo_v3.py \
    --token_dir ${REMOTE_ROOT}/final_sum_large/pretreatment_text/timeset \
    --output_dir ckpts/plus_grpo_v3 \
    --batch_size 1 \
    --lr 3e-6 \
    --max_steps 12000 \
    --save_every 100 \
    --log_every 1 \
    2>&1 | tee ${REMOTE_ROOT}/log_grpo_v3.txt

# ---- 清理 ----
kill $SERVER_PID 2>/dev/null
echo "[Launcher] Done."

