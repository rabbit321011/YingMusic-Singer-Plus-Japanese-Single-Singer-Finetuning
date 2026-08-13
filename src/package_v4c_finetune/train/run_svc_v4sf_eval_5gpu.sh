#!/usr/bin/env bash
set -euo pipefail

ROOT=${REMOTE_ROOT}/final_sum_large_S_v3_20k_d50_test
RUNTIME=${REMOTE_ROOT}/TEMP/s_route_svc_bench/runtime
PYTHON=${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/python
MODEL="$RUNTIME/models/v3_20k_campplus_ft_model.pth"
CONFIG="$RUNTIME/YingMusic-SVC.yml"
GPUS=(0 1 2 3 7)
PIDS=()

cd "$RUNTIME"

mkdir -p "$ROOT/audio" "$ROOT/control/logs" "$ROOT/control/status" \
  "$ROOT/control/failures" "$ROOT/control/scratch"

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for rank in 0 1 2 3 4; do
  gpu=${GPUS[$rank]}
  log="$ROOT/control/logs/worker_${rank}.log"
  echo "LAUNCH rank=$rank physical_gpu=$gpu log=$log"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON" -u "$RUNTIME/run_svc_shard.py" \
      --rank "$rank" \
      --shard "$ROOT/control/shards/shard_${rank}.json" \
      --checkpoint "$MODEL" \
      --config "$CONFIG" \
      --output-dir "$ROOT/audio" \
      --scratch-dir "$ROOT/control/scratch/worker_${rank}" \
      --status "$ROOT/control/status/worker_${rank}.json" \
      --failures "$ROOT/control/failures/worker_${rank}.json" \
      >>"$log" 2>&1 &
  PIDS+=("$!")
done

exit_code=0
while true; do
  alive=0
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  "$PYTHON" "$RUNTIME/summarize_svc_status.py" --status-dir "$ROOT/control/status"
  if [[ "$alive" -eq 0 ]]; then
    break
  fi
  sleep 20
done

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    exit_code=1
  fi
done

"$PYTHON" "$RUNTIME/summarize_svc_status.py" --status-dir "$ROOT/control/status"
if [[ "$exit_code" -ne 0 ]]; then
  echo "V4SF SVC EVAL FINISHED WITH WORKER ERRORS"
  exit "$exit_code"
fi
echo "V4SF SVC EVAL COMPLETE"
