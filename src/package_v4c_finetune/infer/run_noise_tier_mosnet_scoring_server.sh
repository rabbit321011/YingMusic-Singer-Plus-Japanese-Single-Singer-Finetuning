#!/usr/bin/env bash
set -euo pipefail

ROOT=${REMOTE_ROOT}/TEMP/noise_scorer_tiers_20260805
PYTHON="$ROOT/venv_mosnet/bin/python"
SCRIPT="$ROOT/scripts/score_noise_tier_mosnet.py"
PROGRESS="$ROOT/mosnet_scoring_progress.log"
EXIT_FILE="$ROOT/mosnet_scoring.exit"

export CUDA_VISIBLE_DEVICES=-1
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2
export TF_ENABLE_ONEDNN_OPTS=0

rm -f "$EXIT_FILE"
: >"$PROGRESS"
finish() {
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" >"$EXIT_FILE"
  exit "$rc"
}
trap finish EXIT

start=$(date +%s)
printf '[start] %s scorer=mosnet\n' "$(date --iso-8601=seconds)" | tee -a "$PROGRESS"
"$PYTHON" "$SCRIPT" --root "$ROOT" --overwrite \
  2>&1 | tee "$ROOT/scores/mosnet.log"
end=$(date +%s)
printf '[complete] %s scorer=mosnet elapsed=%ss\n' \
  "$(date --iso-8601=seconds)" "$((end - start))" | tee -a "$PROGRESS"

"$PYTHON" - <<'PY'
import csv
from pathlib import Path
path = Path('${REMOTE_ROOT}/TEMP/noise_scorer_tiers_20260805/scores/mosnet.csv')
with path.open(newline='', encoding='utf-8') as handle:
    count = sum(1 for _ in csv.DictReader(handle))
if count != 3054:
    raise RuntimeError(f'mosnet: expected 3054 rows, found {count}')
print(f'[audit] scorer=mosnet rows={count}')
PY
