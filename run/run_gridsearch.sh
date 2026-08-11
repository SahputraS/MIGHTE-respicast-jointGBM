#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

# Usage: ./run/run_gridsearch.sh --target ILI|ARI [extra src/gridsearch_lgbm.py flags]
python3 src/gridsearch_lgbm.py \
  --hub-dir "${ROOT_DIR}/RespiCast-SyndromicIndicators" \
  --data-file "${ROOT_DIR}/data/processed/respicast_long_latest.csv" \
  --gt-file "${ROOT_DIR}/google_data/google_trends_wide.csv" \
  "$@"
