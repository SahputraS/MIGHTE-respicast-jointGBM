#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

# Prospective-only weekly pipeline for RespiCast ILI/ARI.
# Keeps all weekly submissions; does not overwrite existing files.

python3 src/forecast_prospective.py \
  --hub-dir "${ROOT_DIR}/RespiCast-SyndromicIndicators" \
  --targets "ILI,ARI" \
  --model-id "MIGHTE-jointGBM" \
  --canonical-data "${ROOT_DIR}/data/processed/respicast_long_latest.csv" \
  --summary-json "${ROOT_DIR}/data/processed/respicast_long_summary.json" \
  --raw-dir "${ROOT_DIR}/forecasts/prospective/raw" \
  --submission-dir "${ROOT_DIR}/forecasts/prospective/submission" \
  "$@"
