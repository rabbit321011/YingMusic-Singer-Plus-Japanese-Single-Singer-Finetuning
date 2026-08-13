#!/usr/bin/env bash
set -euo pipefail

PROJECT=${REMOTE_ROOT}/YingMusic-Singer-Plus
TRAIN_OUT="$PROJECT/ckpts/v4m_m600d_12k_20260804"
ROOT=${REMOTE_ROOT}/TEMP/v4m_m600d_listen_20260804
PUBLISH="$ROOT/publish"
PYTHON=${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/python3
TORCHRUN=${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/torchrun
EXPORT="$PROJECT/package_v4c_finetune/train/export_v4m_m600d_fsdp_checkpoint.py"
LOG="$ROOT/publish.log"
EXIT="$ROOT/publish.exit"

mkdir -p "$PUBLISH"
rm -f "$EXIT"
on_exit() {
  local rc=$?
  printf '%s\n' "$rc" > "$EXIT"
}
trap on_exit EXIT
exec > >(tee -a "$LOG") 2>&1

printf '[publish start] %s\n' "$(date -Is)"
for STEP in 006000 012000; do
  SOURCE="$TRAIN_OUT/step_${STEP}"
  if [[ "$STEP" == "012000" ]]; then
    SOURCE="${SOURCE}_final"
  fi
  SOURCE="${SOURCE}.dcp"
  OUTPUT="$PUBLISH/V4M_M600D_step_${STEP}.pt"
  if [[ -s "$OUTPUT" && -s "$OUTPUT.sha256" && -s "$OUTPUT.audit.json" ]]; then
    printf '[reuse] %s\n' "$OUTPUT"
    continue
  fi
  printf '[export] %s source=%s\n' "$(date -Is)" "$SOURCE"
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$PROJECT" \
    "$TORCHRUN" --nproc_per_node=4 "$EXPORT" \
      --checkpoint "$SOURCE" \
      --output "$OUTPUT" \
      --config "$PROJECT/src/YingMusicSinger/config/YingMusic_Singer.yaml" \
      --seed 42 \
      --expected_world_size 4
  "$PYTHON" -c "import json,pathlib; p=pathlib.Path('$OUTPUT.audit.json'); x=json.loads(p.read_text()); assert x['global_step']==int('$STEP'); assert x['online_strict_load'] and x['ema_strict_load'] and x['all_finite']; print('[audit ok]',p)"
done

sha256sum "$PUBLISH"/*.pt "$PUBLISH"/*.audit.json | tee "$PUBLISH/SHA256SUMS"
printf '[publish complete] %s\n' "$(date -Is)"
