#!/usr/bin/env bash
set -euo pipefail

ROOT=${REMOTE_ROOT}/TEMP/noise_scorer_tiers_20260805
PYTHON="$ROOT/venv_scorers/bin/python"
SCRIPT="$ROOT/scripts/score_noise_tier_models.py"
PROGRESS="$ROOT/gpu_scoring_progress.log"
EXIT_FILE="$ROOT/gpu_scoring.exit"

export CUDA_VISIBLE_DEVICES=4
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

rm -f "$EXIT_FILE"
: >"$PROGRESS"
finish() {
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" >"$EXIT_FILE"
  exit "$rc"
}
trap finish EXIT

run_scorer() {
  local scorer=$1
  local batch_size=$2
  local workers=$3
  local start end elapsed
  start=$(date +%s)
  printf '[start] %s scorer=%s\n' "$(date --iso-8601=seconds)" "$scorer" \
    | tee -a "$PROGRESS"
  "$PYTHON" "$SCRIPT" \
    --root "$ROOT" \
    --scorer "$scorer" \
    --device cuda \
    --workers "$workers" \
    --batch-size "$batch_size" \
    --overwrite \
    2>&1 | tee "$ROOT/scores/${scorer}.log"
  end=$(date +%s)
  elapsed=$((end - start))
  printf '[complete] %s scorer=%s elapsed=%ss\n' \
    "$(date --iso-8601=seconds)" "$scorer" "$elapsed" | tee -a "$PROGRESS"
}

if [[ "${SKIP_NISQA:-0}" != 1 ]]; then
  run_scorer nisqa 16 4
else
  printf '[skip existing] %s scorer=nisqa\n' "$(date --iso-8601=seconds)" \
    | tee -a "$PROGRESS"
fi
run_scorer singmos_v1 1 0
run_scorer singmos_pro 1 0
run_scorer utmos22 1 0
run_scorer wvmos 1 0
run_scorer utmosv2 8 2

"$PYTHON" - <<'PY'
import csv
from pathlib import Path
root = Path('${REMOTE_ROOT}/TEMP/noise_scorer_tiers_20260805/scores')
for name in ('nisqa', 'singmos_v1', 'singmos_pro', 'utmos22', 'wvmos', 'utmosv2'):
    with (root / f'{name}.csv').open(newline='', encoding='utf-8') as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    if count != 3054:
        raise RuntimeError(f'{name}: expected 3054 rows, found {count}')
    print(f'[audit] scorer={name} rows={count}')
PY

printf '[all complete] %s\n' "$(date --iso-8601=seconds)" | tee -a "$PROGRESS"
