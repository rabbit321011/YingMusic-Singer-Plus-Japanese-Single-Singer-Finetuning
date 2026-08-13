#!/usr/bin/env bash
set -euo pipefail

ROOT=${REMOTE_ROOT}/TEMP/noise_scorer_tiers_20260805
PYTHON="$ROOT/venv_scorers/bin/python"
SCRIPT="$ROOT/scripts/score_noise_tier_models.py"
PROGRESS="$ROOT/cpu_scoring_progress.log"
EXIT_FILE="$ROOT/cpu_scoring.exit"

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
  local workers=$2
  local start end elapsed
  start=$(date +%s)
  printf '[start] %s scorer=%s\n' "$(date --iso-8601=seconds)" "$scorer" \
    | tee -a "$PROGRESS"
  "$PYTHON" "$SCRIPT" \
    --root "$ROOT" \
    --scorer "$scorer" \
    --device cpu \
    --workers "$workers" \
    --batch-size 1 \
    --overwrite \
    2>&1 | tee "$ROOT/scores/${scorer}.log"
  end=$(date +%s)
  elapsed=$((end - start))
  printf '[complete] %s scorer=%s elapsed=%ss\n' \
    "$(date --iso-8601=seconds)" "$scorer" "$elapsed" | tee -a "$PROGRESS"
}

run_scorer dnsmos 8
run_scorer highfreq_reference 8

"$PYTHON" - <<'PY'
import csv
from pathlib import Path
root = Path('${REMOTE_ROOT}/TEMP/noise_scorer_tiers_20260805/scores')
for name in ('dnsmos', 'highfreq_reference'):
    with (root / f'{name}.csv').open(newline='', encoding='utf-8') as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    if count != 3054:
        raise RuntimeError(f'{name}: expected 3054 rows, found {count}')
    print(f'[audit] scorer={name} rows={count}')
PY

printf '[all complete] %s\n' "$(date --iso-8601=seconds)" | tee -a "$PROGRESS"
