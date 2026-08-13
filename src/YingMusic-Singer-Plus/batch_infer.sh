#!/bin/bash
source "${CONDA_SH:-/path/to/conda.sh}"
conda activate yingmusic_plus
cd ${SERVER_ROOT}/YingMusic-Singer-Plus
export CUDA_VISIBLE_DEVICES=4
REF="${REFERENCE_AUDIO:-/path/to/reference.wav}"
MEL="${MELODY_AUDIO:-/path/to/melody.wav}"
for ckpt in ckpts/plus_ja_sft_v4c/step_002000.pt ckpts/plus_ja_sft_v4c/step_004000.pt ckpts/plus_ja_sft_v4c/step_006000.pt ckpts/plus_ja_sft_v4c/step_008000.pt ckpts/plus_ja_sft_v4c/step_010000.pt ckpts/plus_ja_sft_v4c/step_012000.pt ckpts/plus_ja_sft_v4c/step_014000.pt ckpts/plus_ja_sft_v4c/step_016000.pt ckpts/plus_ja_sft_v4c/step_018000.pt ckpts/plus_ja_sft_v4c/step_020000.pt ckpts/plus_ja_sft_v4c/step_022000.pt; do
  step=$(basename $ckpt .pt)
  echo "=== $step ==="
  python infer_v4.py --checkpoint $ckpt --ref_audio "$REF" --melody_audio "$MEL" --ref_text "どこまでも行くよ" --target_text "そう 頑張れないよ 頑張れないよ よそんなんじゃいけないよ アイムレディダーリン 主役はボーリンです セイイエー" --output "test_v4c_${step}_natori.wav" --steps 32 --cfg_strength 3.0 2>&1 | tail -1
done
echo "=== ALL DONE ==="

















