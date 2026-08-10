"""Downloading Google Health Trends timelines per topic/location.

Ported from Download_Gtrends.ipynb. Needs a Google Health Trends API key
with access to the `trends` v1beta API (NOT the same as pytrends).

Set it as an environment variable rather than hardcoding it:
    export GT_API_KEY="your-key-here"
"""

import os
import re
import time
import datetime

import pandas as pd
from googleapiclient.discovery import build

API_VERSION = "v1beta"
DISCOVERY_URL = f"https://www.googleapis.com/discovery/v1/apis/trends/{API_VERSION}/rest"

# label -> topic code (Google Knowledge Graph MID)
ALL_TOPICS = {
    "cough": "/m/01b_21",
    "sore throat": "/m/0b76bty",
    "runny nose": "/m/06p_bp",
    "dyspnea": "/m/01cdt5",
    "influenza": "/m/0cycc",
    "common cold": "/m/0n073",
    "influenza vaccine": "/m/0416v7",
    "nasal congestion": "/m/05s5v6",
    "fever": "/m/0cjf0",
    "paracetamol": "/m/0lbt3",
    "cold medicine": "/m/01nf88",
    "throat lozenge": "/m/08fyrc",
    "influenza A virus": "/m/028tns",
    "avian influenza": "/m/0292d3",
    "flu season": "/m/087cyy",
    "canine influenza": "/m/08pr_0",
    "rapid influenza test": "/m/09gh4jl",
    "influenza B virus": "/m/0b2cnj",
    "virus": "/m/0g9pc",
    "nausea": "/m/0gxb2",
    "headache": "/m/0j5fv",
}

RESPICAST_ILI_URL = "https://raw.githubusercontent.com/european-modelling-hubs/RespiCast-SyndromicIndicators/refs/heads/main/target-data/latest-ILI_incidence.csv"
RESPICAST_ARI_URL = "https://raw.githubusercontent.com/european-modelling-hubs/RespiCast-SyndromicIndicators/refs/heads/main/target-data/latest-ARI_incidence.csv"


def load_hub_locations(exclude: tuple[str, ...] = ("PT",)) -> list[str]:
    """Union of locations present in the RespiCast ILI/ARI target data."""
    ili = pd.read_csv(RESPICAST_ILI_URL)
    ari = pd.read_csv(RESPICAST_ARI_URL)
    locations = set(ili["location"].unique()) | set(ari["location"].unique())
    locations -= set(exclude)
    return sorted(locations)


def slugify(label: str) -> str:
    """'Sore throat' -> 'sore_throat'."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower().strip()).strip("_")


def geo_level_for(loc: str) -> str | None:
    if re.fullmatch(r"[A-Z]{2}", loc):
        return "country"
    if re.fullmatch(r"[A-Z]{2}-[A-Za-z0-9]+", loc):
        return "region"
    if re.fullmatch(r"[0-9]+", loc):
        return "dma"
    return None


def _iso_date(d: str) -> str:
    for fmt in ("%b %d %Y", "%b %Y", "%Y"):
        try:
            return datetime.datetime.strptime(d, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date: {d}")


def get_timeline(
    code: str,
    loc: str,
    level: str,
    api_key: str,
    start_date: str,
    end_date: str,
    frequency: str = "week",
) -> pd.DataFrame:
    """One API call: one topic (or '+'-joined cluster of topics) in one location."""
    service = build(
        "trends", API_VERSION, developerKey=api_key, discoveryServiceUrl=DISCOVERY_URL
    )
    kwargs = dict(
        terms=[code],
        time_startDate=start_date,
        time_endDate=end_date,
        timelineResolution=frequency,
    )
    if level == "country":
        kwargs["geoRestriction_country"] = loc
    elif level == "region":
        kwargs["geoRestriction_region"] = loc
    elif level == "dma":
        kwargs["geoRestriction_dma"] = loc

    res = service.getTimelinesForHealth(**kwargs).execute()
    rows = [
        (_iso_date(pt["date"]), pt["value"])
        for line in res.get("lines", [])
        for pt in line["points"]
    ]
    return pd.DataFrame(rows, columns=["date", "value"]).sort_values("date")


def make_flat_cluster_df(
    locations: list[str], topic_map: dict = ALL_TOPICS
) -> pd.DataFrame:
    """Every (location, keyword) pair as its own unclustered row.

    Feed this to download_all for the initial pass that pulls each keyword
    individually, before gtrends_cluster groups the sparse ones together.
    """
    return pd.DataFrame(
        [
            {"location": loc, "cluster_id": 0, "terms": label}
            for loc in locations
            for label in topic_map
        ]
    )


def download_all(
    cluster_df: pd.DataFrame,
    save_dir: str,
    api_key: str,
    start_date: str,
    end_date: str,
    topic_map: dict = ALL_TOPICS,
    frequency: str = "week",
    sleep_seconds: float = 1.0,
) -> None:
    """Download one CSV per row of cluster_df (location + '+'-joined terms).

    cluster_df needs columns: location, cluster_id, terms (space-separated
    "term1 + term2 + ..."). cluster_id == 0 rows are treated as single,
    unclustered topics (see gtrends_select.make_ok_dict / cluster_for_location
    for how this table is built).
    """
    os.makedirs(save_dir, exist_ok=True)
    print("+++ DOWNLOADING +++")

    for _, row in cluster_df.iterrows():
        loc = row["location"]
        cluster_id = int(row["cluster_id"])
        labels = [t.strip() for t in str(row["terms"]).split(" + ")]

        if cluster_id == 0:
            labels = [labels[0]]
            column_name = labels[0]
            file_name = slugify(labels[0])
        else:
            column_name = "cluster_all"
            file_name = "cluster_all"

        codes = [topic_map[label] for label in labels if label in topic_map]
        missing = [label for label in labels if label not in topic_map]
        for label in missing:
            print(f"  WARNING: no code for '{label}', skipping this term")
        if not codes:
            print(f"  {loc} {column_name}: no valid codes, skipping")
            continue

        level = geo_level_for(loc)
        if level is None:
            print(f"  skip {loc}: can't determine geo level")
            continue

        combined_code = "+".join(codes)
        path = os.path.join(save_dir, f"time_series_{loc}_{file_name}.csv")

        try:
            df = get_timeline(
                combined_code, loc, level, api_key, start_date, end_date, frequency
            )
            df = df.rename(columns={"value": column_name})
            df.to_csv(path, index=False)
            print(f"  saved: {os.path.basename(path)} ({len(labels)} term(s))")
        except Exception as e:
            print(f"  ERROR {loc} {column_name}: {e}")

        time.sleep(sleep_seconds)

    print("+++ DONE +++")
