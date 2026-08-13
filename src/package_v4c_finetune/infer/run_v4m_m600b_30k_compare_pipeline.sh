#!/usr/bin/env bash

set -euo pipefail

PROJECT=${REMOTE_ROOT}/YingMusic-Singer-Plus
INPUT_ROOT=${REMOTE_ROOT}/TEMP/v4m_m600d_listen_20260804
D_ROOT=${REMOTE_ROOT}/TEMP/v4m_m600d_30k_listen_20260806
ROOT=${REMOTE_ROOT}/TEMP/v4m_m600b_30k_compare_20260806
INPUT="$ROOT/input"
SCRIPTS="$INPUT/scripts"
DATASET="$INPUT/dataset"
ALIGNMENT="$INPUT/alignment"
RUNTIME="$INPUT/runtime"
B_DCP="$PROJECT/ckpts/v4m_m600b_30k_20260805/step_030000_final.dcp"
B_PUBLISH="$ROOT/publish/V4M_M600B_step_030000.pt"
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
CLOUD=/TEMP/v4m_m600b_30k_compare_20260806
PACKAGE_NAME=V4M_M600B_30K_compare_named_20260806

mkdir -p "$ROOT/publish" "$SMOKE" "$GENERATED"
if [[ ! -e "$INPUT" ]]; then
    ln -s "$INPUT_ROOT/input" "$INPUT"
fi
for name in HIGHLR_30K_cfg3 HIGHLR_30K_cfg1 M600D_30K_cfg3 M600D_30K_cfg1; do
    if [[ ! -e "$GENERATED/$name" ]]; then
        ln -s "$D_ROOT/generated/$name" "$GENERATED/$name"
    fi
done
rm -f "$EXIT"
on_exit() {
    local rc=$?
    printf '%s\n' "$rc" >"$EXIT"
}
trap on_exit EXIT
exec > >(tee -a "$LOG") 2>&1

printf '[pipeline start] %s\n' "$(date -Is)"
if [[ ! -s "$B_PUBLISH" || ! -s "$B_PUBLISH.audit.json" || ! -s "$B_PUBLISH.sha256" ]]; then
    printf '[export start] %s world=1 device=GPU4\n' "$(date -Is)"
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PROJECT" \
        "$TORCHRUN" --nproc_per_node=1 --master_port=29606 \
        "$PROJECT/package_v4c_finetune/train/export_v4m_m600b_fsdp_checkpoint.py" \
        --checkpoint "$B_DCP" \
        --output "$B_PUBLISH" \
        --config "$PROJECT/src/YingMusicSinger/config/YingMusic_Singer.yaml" \
        --seed 42 \
        --expected_world_size 1
fi
"$PYTHON" -c "import json,pathlib; p=pathlib.Path('$B_PUBLISH.audit.json'); x=json.loads(p.read_text()); assert x['global_step']==30000 and x['online_strict_load'] and x['ema_strict_load'] and x['all_finite'] and x['target_dit_state_elements']==600462048; print('[publish audit ok]', p)"

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
RUNNER="$PROJECT/package_v4c_finetune/infer/v4m_m600b_eval_batch.py"

run_one() {
    local cfg=$1 output=$2 limit=${3:-}
    local extra=()
    if [[ -n "$limit" ]]; then
        extra=(--limit "$limit")
    fi
    local started
    started=$(date +%s)
    printf '[generate] %s label=M600B_30K cfg=%s limit=%s\n' \
        "$(date -Is)" "$cfg" "${limit:-full}"
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PROJECT:$SCRIPTS" \
        "$PYTHON" "$RUNNER" \
        "${COMMON[@]}" \
        --checkpoint "$B_PUBLISH" \
        --output-dir "$output" \
        --cfg "$cfg" \
        "${extra[@]}"
    printf '[elapsed] label=M600B_30K cfg=%s seconds=%s\n' \
        "$cfg" "$(( $(date +%s) - started ))"
}

run_one 3.0 "$SMOKE/M600B_30K_cfg3" 1
"$PYTHON" "$PROJECT/package_v4c_finetune/infer/audit_v4m_m600d_listen_outputs.py" \
    --root "$SMOKE" \
    --dataset "$DATASET" \
    --expected-per-dir 1 \
    --report "$ROOT/smoke_audit.json" \
    --labels M600B_30K \
    --smoke

for cfg in 3.0 1.0; do
    suffix=${cfg%.*}
    output="$GENERATED/M600B_30K_cfg${suffix}"
    run_one "$cfg" "$output"
    test "$(find "$output" -maxdepth 1 -type f -name '*.wav' | wc -l)" -eq 27
    test "$(find "$output/_placement" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 27
done

"$PYTHON" "$PROJECT/package_v4c_finetune/infer/audit_v4m_m600d_listen_outputs.py" \
    --root "$GENERATED" \
    --dataset "$DATASET" \
    --expected-per-dir 27 \
    --report "$ROOT/full_audit.json" \
    --labels HIGHLR_30K M600D_30K M600B_30K

if [[ ! -s "$PACKAGE/package_report.json" ]]; then
    "$PYTHON" "$PROJECT/package_v4c_finetune/infer/build_v4m_m600b_30k_compare_package.py" \
        --root "$ROOT"
fi
"$PYTHON" -c "import json,pathlib; p=pathlib.Path('$PACKAGE/package_report.json'); x=json.loads(p.read_text()); assert x['status']=='ok' and x['generated_wav_count']==162 and x['reference_wav_count']==54 and x['score_rows']==162; print('[package audit ok]', p)"

"$ALIYUNPAN" mkdir "$CLOUD" >/dev/null 2>&1 || true
"$ALIYUNPAN" upload -bs 30720 \
    "$PACKAGE/$PACKAGE_NAME.zip" \
    "$PACKAGE/$PACKAGE_NAME.zip.sha256" \
    "$PACKAGE/package_report.json" \
    "$CLOUD"
"$ALIYUNPAN" ls "$CLOUD"
printf '[pipeline complete] %s\n' "$(date -Is)"
