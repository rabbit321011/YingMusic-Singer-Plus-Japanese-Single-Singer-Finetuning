#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${V4IJPH_RESUME_GATE_ROOT:-${REMOTE_ROOT}/TEMP/v4ijph_v2_resume_gate}"
CONTINUOUS="$ROOT/continuous"
SPLIT="$ROOT/split"
SOURCE="$SCRIPT_DIR/run_v4ijph_30k_ddp.sh"
AUDITOR="$SCRIPT_DIR/audit_v4ijph_checkpoint.py"
COMPARATOR="$SCRIPT_DIR/compare_v4ijph_checkpoints.py"
PYTHON="${V4IJPH_PYTHON:-${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/python}"
V4IPH="${REMOTE_ROOT}/YingMusic-Singer-Plus/ckpts/plus_ja_sft_v4iph/step_008000.pt"
V4IPH_SHA="ddfd35941df398619349cc18597e3a63e4fb52019145476eb8f8b2c91dc15e10"

for directory in "$CONTINUOUS" "$SPLIT"; do
    if [[ -e "$directory" ]]; then
        echo "Resume gate directory already exists; choose a new V4IJPH_RESUME_GATE_ROOT: $directory" >&2
        exit 2
    fi
done

V4IJPH_OUTPUT_DIR="$CONTINUOUS" bash "$SOURCE" \
    --stop_after_step 8010 --save_every 2000 --eval_every 1000 --log_every 5

V4IJPH_OUTPUT_DIR="$SPLIT" bash "$SOURCE" \
    --stop_after_step 8005 --save_every 2000 --eval_every 1000 --log_every 5

V4IJPH_OUTPUT_DIR="$SPLIT" bash "$SOURCE" \
    --resume "$SPLIT/step_008005_final.pt" \
    --stop_after_step 8010 --save_every 2000 --eval_every 1000 --log_every 5

for checkpoint in \
    "$CONTINUOUS/step_008010_final.pt" \
    "$SPLIT/step_008010_final.pt"; do
    "$PYTHON" "$AUDITOR" "$checkpoint" \
        --expected-step 8010 \
        --expected-run-state stopped \
        --source-v4iph-checkpoint "$V4IPH" \
        --source-v4iph-sha256 "$V4IPH_SHA"
done

"$PYTHON" "$COMPARATOR" \
    "$CONTINUOUS/step_008010_final.pt" \
    "$SPLIT/step_008010_final.pt" \
    --expected-step 8010
