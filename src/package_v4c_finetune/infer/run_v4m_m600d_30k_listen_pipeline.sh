#!/usr/bin/env bash

set -euo pipefail

PROJECT=${REMOTE_ROOT}/YingMusic-Singer-Plus
OLD_ROOT=${REMOTE_ROOT}/TEMP/v4m_m600d_listen_20260804
ROOT=${REMOTE_ROOT}/TEMP/v4m_m600d_30k_listen_20260806
INPUT="$ROOT/input"
SCRIPTS="$INPUT/scripts"
DATASET="$INPUT/dataset"
ALIGNMENT="$INPUT/alignment"
RUNTIME="$INPUT/runtime"
PUBLISH="$ROOT/publish/V4M_M600D_step_030000.pt"
DCP="$PROJECT/ckpts/v4m_m600d_12k_20260804/step_030000_final.dcp"
HIGHLR="$PROJECT/ckpts/plus_ja_sft_v4ph_30k_highlr/step_030000_final.pt"
PYTHON=${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/python3
TORCHRUN=${REMOTE_ROOT}/.conda/envs/yingmusic_plus/bin/torchrun
VAE="$PROJECT/ckpts/stable_audio_2_0_vae_20hz_official.ckpt"
GAME_REPO=${REMOTE_ROOT}/TEMP/v4timetest_20260802/OpenVPI-GAME-full
GAME_DEPS=${REMOTE_ROOT}/.conda/envs/yingmusic_plus/lib/python3.10/site-packages
GAME_MODEL=${REMOTE_ROOT}/TEMP/v4timetest_20260802/GAME-1.0-medium/GAME-1.0-medium/model.pt
GAME_MANIFEST=${REMOTE_ROOT}/TEMP/v4ph_game_cache_full_20260730_v2/manifest.json
SMOKE="$ROOT/smoke"
GENERATED="$ROOT/generated"
PACKAGE="$ROOT/package"
LOG="$ROOT/pipeline.log"
EXIT="$ROOT/pipeline.exit"
ALIYUNPAN=${REMOTE_ROOT}/bin/aliyunpan
CLOUD=/TEMP/v4m_m600d_30k_listen_20260806

mkdir -p "$ROOT/publish" "$SMOKE" "$GENERATED"
if [[ ! -e "$INPUT" ]]; then
    ln -s "$OLD_ROOT/input" "$INPUT"
fi
rm -f "$EXIT"
on_exit() {
    local rc=$?
    printf '%s\n' "$rc" >"$EXIT"
}
trap on_exit EXIT
exec > >(tee -a "$LOG") 2>&1

printf '[pipeline start] %s\n' "$(date -Is)"
if [[ ! -s "$PUBLISH" || ! -s "$PUBLISH.audit.json" || ! -s "$PUBLISH.sha256" ]]; then
    printf '[export start] %s world=1 device=GPU4\n' "$(date -Is)"
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PROJECT" \
        "$TORCHRUN" --nproc_per_node=1 --master_port=29605 \
        "$PROJECT/package_v4c_finetune/train/export_v4m_m600d_fsdp_checkpoint.py" \
        --checkpoint "$DCP" \
        --output "$PUBLISH" \
        --config "$PROJECT/src/YingMusicSinger/config/YingMusic_Singer.yaml" \
        --seed 42 \
        --expected_world_size 1
fi
"$PYTHON" -c "import json,pathlib; p=pathlib.Path('$PUBLISH.audit.json'); x=json.loads(p.read_text()); assert x['global_step']==30000 and x['online_strict_load'] and x['ema_strict_load'] and x['all_finite'] and x['target_dit_state_elements']==602599648; print('[publish audit ok]', p)"

COMMON=(
    --dataset "$DATASET"
    --alignment-dir "$ALIGNMENT"
    --runtime "$RUNTIME"
    --singer-root "$PROJECT"
    --vae-ckpt "$VAE"
    --game-repo "$GAME_REPO"
    --game-deps "$GAME_DEPS"
    --game-model "$GAME_MODEL"
    --game-cache-manifest "$GAME_MANIFEST"
    --steps 32
    --seed 42
    --device cuda:0
    --resume
)

run_one() {
    local runner=$1 checkpoint=$2 label=$3 cfg=$4 output=$5 limit=${6:-}
    local extra=()
    if [[ -n "$limit" ]]; then
        extra=(--limit "$limit")
    fi
    local started
    started=$(date +%s)
    printf '[generate] %s label=%s cfg=%s limit=%s\n' \
        "$(date -Is)" "$label" "$cfg" "${limit:-full}"
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PROJECT:$SCRIPTS" \
        "$PYTHON" "$runner" \
        "${COMMON[@]}" \
        --checkpoint "$checkpoint" \
        --output-dir "$output" \
        --cfg "$cfg" \
        "${extra[@]}"
    printf '[elapsed] label=%s cfg=%s seconds=%s\n' \
        "$label" "$cfg" "$(( $(date +%s) - started ))"
}

RUNNERS=(
    "$SCRIPTS/v4ph_eval_batch.py"
    "$PROJECT/package_v4c_finetune/infer/v4m_m600d_eval_batch.py"
)
CHECKPOINTS=("$HIGHLR" "$PUBLISH")
LABELS=("HIGHLR_30K" "M600D_30K")

printf '[smoke start] %s\n' "$(date -Is)"
for index in 0 1; do
    run_one "${RUNNERS[$index]}" "${CHECKPOINTS[$index]}" \
        "${LABELS[$index]}" 3.0 "$SMOKE/${LABELS[$index]}_cfg3" 1
done
"$PYTHON" "$PROJECT/package_v4c_finetune/infer/audit_v4m_m600d_listen_outputs.py" \
    --root "$SMOKE" \
    --dataset "$DATASET" \
    --expected-per-dir 1 \
    --report "$ROOT/smoke_audit.json" \
    --labels "${LABELS[@]}" \
    --smoke

printf '[full start] %s\n' "$(date -Is)"
for index in 0 1; do
    for cfg in 3.0 1.0; do
        suffix=${cfg%.*}
        output="$GENERATED/${LABELS[$index]}_cfg${suffix}"
        run_one "${RUNNERS[$index]}" "${CHECKPOINTS[$index]}" \
            "${LABELS[$index]}" "$cfg" "$output"
        test "$(find "$output" -maxdepth 1 -type f -name '*.wav' | wc -l)" -eq 27
        test "$(find "$output/_placement" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 27
    done
done
"$PYTHON" "$PROJECT/package_v4c_finetune/infer/audit_v4m_m600d_listen_outputs.py" \
    --root "$GENERATED" \
    --dataset "$DATASET" \
    --expected-per-dir 27 \
    --report "$ROOT/full_audit.json" \
    --labels "${LABELS[@]}"

if [[ ! -s "$PACKAGE/package_report.json" ]]; then
    "$PYTHON" "$PROJECT/package_v4c_finetune/infer/build_v4m_m600d_30k_listen_package.py" \
        --root "$ROOT"
fi
"$PYTHON" -c "import json,pathlib; p=pathlib.Path('$PACKAGE/package_report.json'); x=json.loads(p.read_text()); assert x['status']=='ok' and x['wav_count']==108 and x['score_rows']==108; print('[package audit ok]', p)"

"$ALIYUNPAN" mkdir "$CLOUD" >/dev/null 2>&1 || true
printf '[upload start] %s\n' "$(date -Is)"
"$ALIYUNPAN" upload -bs 30720 \
    "$PACKAGE/V4M_M600D_30K_blind_20260806.zip" \
    "$PACKAGE/V4M_M600D_30K_blind_20260806.zip.sha256" \
    "$PACKAGE/V4M_M600D_30K_technical_20260806.zip" \
    "$PACKAGE/package_report.json" \
    "$CLOUD"
"$ALIYUNPAN" ls "$CLOUD"
printf '[pipeline complete] %s\n' "$(date -Is)"
