#!/bin/bash
set -euo pipefail

cd ${REMOTE_ROOT}/YingMusic-Singer-Plus
TARGET="/TEMP/v4vf_lr1e4_30k"
LOG="${REMOTE_ROOT}/TEMP/v4vf_cloud_upload.log"

${REMOTE_ROOT}/bin/aliyunpan mkdir "$TARGET" >/dev/null 2>&1 || true
echo "START $(date -Is)" | tee "$LOG"
${REMOTE_ROOT}/bin/aliyunpan upload -bs 30720 \
    ckpts/plus_ja_sft_v4vf_lr1e4_30k/step_012000.pt \
    ckpts/plus_ja_sft_v4vf_lr1e4_30k/step_024000.pt \
    ckpts/plus_ja_sft_v4vf_lr1e4_30k/step_030000.pt \
    "$TARGET" 2>&1 | tee -a "$LOG"
echo "EXIT ${PIPESTATUS[0]} $(date -Is)" | tee -a "$LOG"
