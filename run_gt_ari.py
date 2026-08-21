#!/usr/bin/env python3

import json
import os
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path


# MIGHTE_PROJECT_ROOT lets this run locally for testing; unset (the normal case on
# the server) it falls back to the hardcoded server path unchanged.
PROJECT = Path(os.environ.get("MIGHTE_PROJECT_ROOT", "/data/shared/nsahputra/projects/MIGHTE-respicast-jointGBM"))

# Season-holdout-safe gridsearch output (see Season_Holdout_Gridsearch.ipynb): one
# file per target/variant, single top-level key matching the variant name.
PARAMS_JSON = PROJECT / "best_params_season2025_26_holdout_ari_gt_proc.json"
HUB_DIR = PROJECT / "RespiCast-SyndromicIndicators"

CANONICAL_DATA = PROJECT / "data" / "processed" / "respicast_long_latest.csv"
SUMMARY_JSON = PROJECT / "data" / "processed" / "respicast_long_summary.json"

# Season-holdout-safe GT file (denoise/detrend fit only on pre-season data -- see
# google_preprocessing/notebooks/main_season_holdout.ipynb). NOT the operational
# google_trends_preprocessed.csv, which would leak 2025/26-and-beyond info into
# every pre-season feature value.
GOOGLE_TRENDS_FILE = PROJECT / "google_preprocessing" / "data" / "processed" / "google_trends_preprocessed_season_holdout.csv"

OUTPUT_DIR = Path(os.environ.get(
    "MIGHTE_OUTPUT_ROOT", "/data/shared/nsahputra/outputs"
)) / "MIGHTE-ISI_lgbm_google_ari(25bags)"

# Threads per LightGBM fit -- keep this at 1 on a quota-limited shared server (e.g. a
# 0.1 CPU allocation). Leaving LightGBM at its own "use all available cores" default
# there crashes with "libgomp: Thread creation failed: Resource temporarily
# unavailable" the moment it tries to spawn more threads than the cgroup allows.
NUM_THREADS = os.environ.get("MIGHTE_NUM_THREADS", "1")


def require_exists(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


# server setup: do not accidentally use all CPU cores on shared server
# (was hardcoded to "100" -- almost certainly wrong on a quota-limited box; use
# NUM_THREADS above, which also now gets passed through to LightGBM itself via
# --num-threads below, not just these BLAS/OpenMP env vars)
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS


require_exists(PROJECT, "Project folder")
require_exists(PARAMS_JSON, "Best params JSON")
require_exists(HUB_DIR, "RespiCast-SyndromicIndicators repo")
require_exists(CANONICAL_DATA, "Canonical data")
require_exists(SUMMARY_JSON, "Summary JSON")
require_exists(GOOGLE_TRENDS_FILE, "Google Trends file")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(PARAMS_JSON, "r") as f:
    best = json.load(f)

print("Loaded configs:", list(best.keys()))

cfg = best["gt_proc"]

NUM_LEAVES = cfg["num_leaves"]
LEARNING_RATE = cfg["learning_rate"]
MIN_CHILD_SAMPLES = cfg["min_child_samples"]
FEATURE_FRACTION = cfg["feature_fraction"]
STAGE1_ROUNDS = cfg["rounds"]
STAGE2_ROUNDS = cfg["stage2_rounds"]
LAMBDA_L2 = cfg["lambda_l2"]
S2_MIN_CHILD_SAMPLES = cfg["s2_min_child_samples"]

START_ORIGIN = date.fromisocalendar(2025, 40, 7).isoformat()

cmd = [
    sys.executable,
    str(PROJECT / "src" / "forecast_backtest.py"),

    "--hub-dir", str(HUB_DIR),
    "--targets", "ARI",
    "--start-origin-date", START_ORIGIN,
    "--submission-dir", str(OUTPUT_DIR),
    "--snapshots-only",

    "--num-bags", "25",
    "--location-bag-frac", "0.8",

    "--canonical-data", str(CANONICAL_DATA),
    "--summary-json", str(SUMMARY_JSON),
    "--google-trends-file", str(GOOGLE_TRENDS_FILE),

    "--num-leaves", str(NUM_LEAVES),
    "--learning-rate", str(LEARNING_RATE),
    "--min-child-samples", str(MIN_CHILD_SAMPLES),
    "--feature-fraction", str(FEATURE_FRACTION),
    "--stage1-rounds", str(STAGE1_ROUNDS),
    "--stage2-rounds", str(STAGE2_ROUNDS),
    "--lambda-l2", str(LAMBDA_L2),
    "--s2-min-child-samples", str(S2_MIN_CHILD_SAMPLES),

    "--num-threads", NUM_THREADS,

    "--exclude-covid",
]

print("Running GT processed LightGBM forecast")
print("Project:", PROJECT)
print("Params:", PARAMS_JSON)
print("Hub dir:", HUB_DIR)
print("Canonical data:", CANONICAL_DATA)
print("Summary JSON:", SUMMARY_JSON)
print("Google Trends:", GOOGLE_TRENDS_FILE)
print("Start origin:", START_ORIGIN)
print("Output dir:", OUTPUT_DIR)
print("Threads:", NUM_THREADS)
print()
print("Command:")
print(" ".join(shlex.quote(x) for x in cmd))
print()

subprocess.run(cmd, cwd=PROJECT, check=True)