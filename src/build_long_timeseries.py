#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from io_respicast import discover_target_files, load_locations_map, read_target_file, source_priority_map


def _raw_frame(hub_dir: Path) -> pd.DataFrame:
    target_data_dir = hub_dir / "target-data"
    specs = discover_target_files(target_data_dir)
    if not specs:
        raise ValueError(f"No target files found under {target_data_dir}")

    frames = [read_target_file(spec) for spec in specs]
    raw = pd.concat(frames, ignore_index=True)

    # De-duplicate exact key repeats inside a single file by averaging value.
    raw = (
        raw.groupby(
            ["target", "location", "truth_date", "year_week", "source", "file_kind", "file_path", "snapshot_date"],
            as_index=False,
        )["value"]
        .mean()
    )
    return raw


def _select_canonical(raw: pd.DataFrame, hub_dir: Path) -> pd.DataFrame:
    raw = raw.copy()
    src_rank: Dict[str, int] = source_priority_map()
    raw["source_priority"] = raw["source"].map(src_rank).fillna(999).astype(int)
    raw["is_latest"] = (raw["file_kind"] == "latest").astype(int)

    # Precedence rule:
    # 1) latest-* files
    # 2) newer snapshots
    # 3) source priority (ERVISS before FluID)
    raw_sorted = raw.sort_values(
        ["target", "location", "truth_date", "is_latest", "snapshot_date", "source_priority"],
        ascending=[True, True, True, False, False, True],
        kind="mergesort",
    )

    selected = raw_sorted.drop_duplicates(subset=["target", "location", "truth_date"], keep="first").copy()

    loc_map_path = hub_dir / "supporting-files" / "locations_iso2_codes.csv"
    if loc_map_path.exists():
        loc_map = load_locations_map(loc_map_path)
        selected = selected.merge(loc_map, on="location", how="left")
    else:
        selected["location_name"] = pd.NA

    selected["truth_date"] = pd.to_datetime(selected["truth_date"]).dt.date
    selected = selected.sort_values(["target", "location", "truth_date"]).reset_index(drop=True)

    final_cols = [
        "target",
        "location",
        "location_name",
        "truth_date",
        "year_week",
        "value",
        "source",
        "file_kind",
        "snapshot_date",
        "file_path",
    ]
    for col in final_cols:
        if col not in selected.columns:
            selected[col] = pd.NA

    return selected.loc[:, final_cols]


def resolve_long_timeseries(hub_dir: Path) -> pd.DataFrame:
    """Canonical series reflecting the current, most-revised knowledge of every
    week — latest-*.csv rows win unconditionally over any snapshot, regardless
    of how old or new. Right for prospective forecasting (you want the best
    known value for every past week). NOT vintage-faithful for backtesting —
    see resolve_long_timeseries_asof for that."""
    raw = _raw_frame(hub_dir)
    return _select_canonical(raw, hub_dir)


def resolve_long_timeseries_asof(hub_dir: Path, as_of_date) -> pd.DataFrame:
    """Vintage-aware canonical series: only uses data that would genuinely have
    been available by `as_of_date`.

    latest-*.csv files are excluded entirely — they carry no date of their own
    and always reflect "now", so including them would reintroduce exactly the
    look-ahead bias this function exists to avoid (e.g. a February-origin
    backtest run today silently training on February data that's since been
    revised/backfilled well past what a real forecaster had in February).
    Only dated snapshot files with snapshot_date <= as_of_date are used; for
    each (target, location, truth_date), the newest such snapshot wins, with
    source priority (ERVISS before FluID) breaking remaining ties. A week with
    no qualifying snapshot is simply absent, matching what a real forecaster
    would have faced.
    """
    as_of = pd.to_datetime(as_of_date)
    raw = _raw_frame(hub_dir)
    raw = raw[(raw["file_kind"] == "snapshot") & (raw["snapshot_date"] <= as_of)].copy()
    if raw.empty:
        raise ValueError(f"No snapshot data available as of {as_of.date()} under {hub_dir}")
    return _select_canonical(raw, hub_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a canonical long RespiCast time series from latest + snapshot files.")
    parser.add_argument(
        "--hub-dir",
        default="RespiCast-SyndromicIndicators",
        help="Path to RespiCast-SyndromicIndicators directory",
    )
    parser.add_argument(
        "--output",
        default="data/processed/respicast_long_latest.csv",
        help="Output canonical long-series CSV path",
    )
    parser.add_argument(
        "--summary-json",
        default="data/processed/respicast_long_summary.json",
        help="Summary diagnostics JSON path",
    )
    args = parser.parse_args()

    hub_dir = Path(args.hub_dir).resolve()
    output_path = Path(args.output).resolve()
    summary_path = Path(args.summary_json).resolve()

    canonical = resolve_long_timeseries(hub_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(output_path, index=False)

    summary = {
        "rows": int(len(canonical)),
        "targets": sorted(canonical["target"].dropna().unique().tolist()),
        "locations": int(canonical["location"].nunique()),
        "date_min": str(canonical["truth_date"].min()),
        "date_max": str(canonical["truth_date"].max()),
        "rows_by_target": canonical.groupby("target").size().astype(int).to_dict(),
        "rows_by_source": canonical.groupby("source").size().astype(int).to_dict(),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved canonical series: {output_path} ({len(canonical)} rows)")
    print(f"Saved summary: {summary_path}")
    print(f"Targets: {', '.join(summary['targets'])}")
    print(f"Date range: {summary['date_min']} to {summary['date_max']}")


if __name__ == "__main__":
    main()
