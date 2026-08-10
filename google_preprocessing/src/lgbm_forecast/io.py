"""Reading the raw per-location Google Trends CSVs into one wide DataFrame."""

import os
import re
from datetime import date

import pandas as pd


def load_timeseries_wide(
    folder_path: str,
    locations: list[str],
    only_clusters: bool = False,
) -> pd.DataFrame:
    """Merge each location's `time_series_{loc}_*.csv` files into one wide row per date.

    Set only_clusters=True to load just the `..._cluster_all.csv` files
    produced by gtrends_download.download_all for CLUSTER columns; the
    default (False) loads the individual-keyword files and skips clusters.
    """
    frames = []

    for loc in locations:
        if only_clusters:
            pattern = re.compile(rf"time_series_{re.escape(loc)}_cluster_all\.csv$")
            matched_files = [f for f in os.listdir(folder_path) if pattern.match(f)]
        else:
            matched_files = [
                f
                for f in os.listdir(folder_path)
                if re.match(rf"time_series_{re.escape(loc)}_.*\.csv", f)
                and not re.search(r"_\d+\.csv$", f)
                and "_cluster" not in f
            ]

        if not matched_files:
            continue

        df_loc = None
        for fname in matched_files:
            df_tmp = pd.read_csv(os.path.join(folder_path, fname))
            df_loc = df_tmp if df_loc is None else df_loc.merge(df_tmp, on="date", how="outer")

        df_loc["location"] = loc
        frames.append(df_loc)

    if not frames:
        return pd.DataFrame(columns=["date", "location"])

    return pd.concat(frames, ignore_index=True)


def split_train_test(
    df: pd.DataFrame,
    train_start: str = "2014-10-01",
    train_end_iso: tuple[int, int] = (2025, 40),
    test_start_iso: tuple[int, int] = (2025, 20),
    test_end_iso: tuple[int, int] = (2026, 20),
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test using ISO year-week boundaries.

    Test starts 20 weeks before the train cutoff on purpose: that overlap
    is the warm-up window the denoiser (WINDOW=20) needs to produce its
    first non-NaN value at the true start of the test period.
    """
    max_train = str(date.fromisocalendar(*train_end_iso, 7))
    start_test = str(date.fromisocalendar(*test_start_iso, 7))
    end_test = str(date.fromisocalendar(*test_end_iso, 7))

    test = df[(df[date_col] >= start_test) & (df[date_col] <= end_test)].copy()
    train = df[(df[date_col] > train_start) & (df[date_col] < max_train)].copy()
    return train, test
