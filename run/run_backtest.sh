#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

python3 src/forecast_backtest.py \
  --hub-dir "${ROOT_DIR}/RespiCast-SyndromicIndicators" \
  --targets "ILI,ARI" \
  --model-id "MIGHTE-jointGBM" \
  --start-origin-date "2025-10-01" \
  --canonical-data "${ROOT_DIR}/data/processed/respicast_long_latest.csv" \
  --summary-json "${ROOT_DIR}/data/processed/respicast_long_summary.json" \
  --raw-dir "${ROOT_DIR}/forecasts/retrospective/raw" \
  --submission-dir "${ROOT_DIR}/forecasts/retrospective/submission" \
  "$@"
