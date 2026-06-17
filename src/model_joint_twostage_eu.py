#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

from distributions import GaussianFrozenLoc, GaussianFrozenLocBounded

try:
    from lightgbmlss.model import LightGBMLSS
except ImportError as exc:
    raise ImportError("lightgbmlss is required for model_joint_twostage_eu.py") from exc


QUANTILES = np.array(
    [
        0.01,
        0.025,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        0.975,
        0.99,
    ],
    dtype=float,
)


@dataclass
class RuntimeConfig:
    data_file: Path
    target: str
    output: Path
    locations_file: Path
    forecasting_weeks_file: Path
    max_horizons: int
    num_bags: int
    bag_frac: float
    location_bag_frac: float
    location_bag_min: int
    seed: int
    stage1_rounds: int
    stage2_rounds: int
    own_lags: List[int]
    donor_lags: List[int]
    donor_top_k: int
    other_top_k: int
    min_overlap: int
    min_train_rows: int
    target_mode: str
    sigma_mode: str
    locations_subset: Optional[List[str]] = None
    recent_weeks_required: int = 4
    anchor_date: Optional[pd.Timestamp] = None
    origin_date: Optional[pd.Timestamp] = None


def parse_lag_string(lag_str: str) -> List[int]:
    vals: List[int] = []
    for token in lag_str.split(","):
        token = token.strip()
        if not token:
            continue
        lag = int(token)
        if lag < 1:
            raise ValueError("All lags must be >= 1")
        vals.append(lag)
    if not vals:
        raise ValueError("At least one lag must be specified")
    return sorted(set(vals))


def infer_flu_season(ts: pd.Timestamp) -> str:
    start_year = ts.year if ts.month >= 10 else ts.year - 1
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def _load_target_panel(
    data_file: Path,
    target: str,
    locations: Sequence[str],
    cutoff_date: Optional[pd.Timestamp] = None,
    calendar_end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    df = pd.read_csv(data_file)
    required = {"target", "location", "truth_date", "value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{data_file} missing required columns: {sorted(missing)}")

    df = df[df["target"] == target].copy()
    df["location"] = df["location"].astype(str)
    df["truth_date"] = pd.to_datetime(df["truth_date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["location", "truth_date", "value"])
    df = df[df["location"].isin(set(locations))].copy()
    if cutoff_date is not None:
        cutoff = pd.to_datetime(cutoff_date)
        df = df[df["truth_date"] <= cutoff].copy()

    if df.empty:
        raise ValueError(f"No rows remain for target '{target}' after location filtering")

    # Average duplicates per location/date.
    df = (
        df.groupby(["location", "truth_date"], as_index=False)["value"]
        .mean()
        .sort_values(["location", "truth_date"])
    )

    # Weekly calendar grid (Sunday week-end dates in this hub).
    cal_end = pd.to_datetime(calendar_end_date) if calendar_end_date is not None else df["truth_date"].max()
    all_dates = pd.date_range(df["truth_date"].min(), cal_end, freq="W-SUN")
    grid = pd.MultiIndex.from_product([locations, all_dates], names=["location", "truth_date"]).to_frame(index=False)
    out = grid.merge(df, on=["location", "truth_date"], how="left")
    out = out.sort_values(["location", "truth_date"]).reset_index(drop=True)
    out = out.rename(columns={"truth_date": "date", "value": "y"})
    return out


def _pivot(df_long: pd.DataFrame) -> pd.DataFrame:
    return df_long.pivot(index="date", columns="location", values="y").sort_index()


def _series_for_loc(pivot: pd.DataFrame, loc: str) -> pd.Series:
    if loc not in pivot.columns:
        return pd.Series(np.nan, index=pivot.index)
    return pivot[loc]


def compute_top_donors(
    target_pivot: pd.DataFrame,
    candidate_pivot: pd.DataFrame,
    top_k: int,
    min_overlap: int,
    target_mode: str,
) -> Dict[str, List[str]]:
    if top_k <= 0:
        return {}

    tgt_log = np.log1p(np.clip(target_pivot.astype(float), 0.0, None))
    cand_log = np.log1p(np.clip(candidate_pivot.astype(float), 0.0, None))

    if target_mode == "delta_log":
        tgt_score = tgt_log.diff(1)
        cand_score = cand_log.diff(1)
    else:
        tgt_score = tgt_log
        cand_score = cand_log

    out: Dict[str, List[str]] = {}
    for loc in target_pivot.columns:
        target = tgt_score[loc]
        pairs: List[Tuple[str, float]] = []
        for cand in candidate_pivot.columns:
            if cand == loc and target_pivot is candidate_pivot:
                continue
            pair = pd.concat([target, cand_score[cand]], axis=1).dropna()
            if len(pair) < min_overlap:
                continue
            a_std = float(pair.iloc[:, 0].std())
            b_std = float(pair.iloc[:, 1].std())
            if a_std <= 1e-12 or b_std <= 1e-12:
                continue
            corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
            if np.isnan(corr):
                continue
            pairs.append((cand, corr))
        pairs.sort(key=lambda x: x[1], reverse=True)
        out[loc] = [name for name, _ in pairs[:top_k]]
    return out


def build_features(
    target_df: pd.DataFrame,
    other_df: pd.DataFrame,
    own_lags: Sequence[int],
    donor_lags: Sequence[int],
    same_target_donors: Dict[str, List[str]],
    other_target_donors: Dict[str, List[str]],
    donor_top_k: int,
    other_top_k: int,
) -> pd.DataFrame:
    df = target_df.copy().sort_values(["location", "date"]).reset_index(drop=True)
    g = df.groupby("location", group_keys=False)

    # Last observed level up to date t (gap-tolerant baseline for delta modeling).
    df["y_base"] = g["y"].ffill()

    for lag in own_lags:
        df[f"y_lag_{lag}"] = g["y"].shift(lag)

    df["y_diff_1"] = g["y"].diff(1)
    df["y_diff_4"] = g["y"].diff(4)
    df["y_pct_chg_1"] = g["y"].pct_change(1, fill_method=None).replace([np.inf, -np.inf], np.nan)

    # Seasonal time basis.
    woy = df["date"].dt.isocalendar().week.astype(float)
    df["week_sin"] = np.sin(2 * np.pi * woy / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * woy / 52.0)
    df["is_flu_season"] = ((df["date"].dt.month >= 10) | (df["date"].dt.month <= 3)).astype(int)

    tgt_pivot = _pivot(target_df)
    oth_pivot = _pivot(other_df)

    # Global target and opposite-target covariates.
    nat_mean_tgt = tgt_pivot.mean(axis=1)
    nat_std_tgt = tgt_pivot.std(axis=1)
    nat_mean_oth = oth_pivot.mean(axis=1)

    global_df = pd.DataFrame({"date": tgt_pivot.index})
    global_df["nat_mean_tgt_lag_1"] = nat_mean_tgt.shift(1)
    global_df["nat_mean_tgt_lag_4"] = nat_mean_tgt.shift(4)
    global_df["nat_std_tgt_lag_1"] = nat_std_tgt.shift(1)
    global_df["nat_mean_oth_lag_1"] = nat_mean_oth.shift(1)
    global_df["nat_mean_oth_lag_4"] = nat_mean_oth.shift(4)

    df = df.merge(global_df, on="date", how="left")

    # Opposite target, same location lags.
    other_loc_value = other_df.rename(columns={"y": "other_y"})
    df = df.merge(other_loc_value[["location", "date", "other_y"]], on=["location", "date"], how="left")
    g2 = df.groupby("location", group_keys=False)
    for lag in donor_lags:
        df[f"other_loc_lag_{lag}"] = g2["other_y"].shift(lag)

    # Donor features from same target.
    for loc, grp in df.groupby("location", sort=False):
        donors = same_target_donors.get(loc, [])
        idx = grp.index
        for rank in range(1, donor_top_k + 1):
            donor = donors[rank - 1] if rank - 1 < len(donors) else None
            for lag in donor_lags:
                col = f"donor_tgt_{rank}_lag_{lag}"
                if donor is None:
                    df.loc[idx, col] = np.nan
                else:
                    df.loc[idx, col] = _series_for_loc(tgt_pivot, donor).shift(lag).reindex(grp["date"]).to_numpy()

    # Donor features from opposite target.
    for loc, grp in df.groupby("location", sort=False):
        donors = other_target_donors.get(loc, [])
        idx = grp.index
        for rank in range(1, other_top_k + 1):
            donor = donors[rank - 1] if rank - 1 < len(donors) else None
            for lag in donor_lags:
                col = f"donor_oth_{rank}_lag_{lag}"
                if donor is None:
                    df.loc[idx, col] = np.nan
                else:
                    df.loc[idx, col] = _series_for_loc(oth_pivot, donor).shift(lag).reindex(grp["date"]).to_numpy()

    # Location identity (joint model).
    loc_dummies = pd.get_dummies(df["location"], prefix="loc", dtype=float)
    df = pd.concat([df, loc_dummies], axis=1)

    df["season"] = df["date"].map(infer_flu_season)
    return df


def build_pooled_examples(feature_df: pd.DataFrame, max_horizons: int) -> pd.DataFrame:
    parts = []
    for h in range(1, max_horizons + 1):
        # Forecast horizons are strictly ahead of the last observed anchor week.
        # h=1 => anchor+1 week, ..., h=4 => anchor+4 weeks.
        shift_weeks = h
        tmp = feature_df.copy()
        tmp["horizon"] = h
        tmp["target"] = tmp.groupby("location")["y"].shift(-shift_weeks)
        tmp["target_end_date"] = tmp["date"] + pd.Timedelta(weeks=shift_weeks)
        tmp["horizon_sin"] = np.sin(2 * np.pi * h / 8.0)
        tmp["horizon_cos"] = np.cos(2 * np.pi * h / 8.0)
        parts.append(tmp)
    return pd.concat(parts, axis=0, ignore_index=True)


def feature_columns(pooled_df: pd.DataFrame) -> List[str]:
    excluded = {"location", "date", "y", "target", "target_end_date", "season", "other_y", "y_base"}
    return [c for c in pooled_df.columns if c not in excluded]


def distribution_for_mode(mode: str):
    if mode == "bounded":
        return GaussianFrozenLocBounded(sigma_min=0.1, sigma_max=0.6)
    if mode == "unbounded":
        return GaussianFrozenLoc()
    raise ValueError(f"Unsupported sigma mode: {mode}")


def fit_two_stage_one_bag(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    stage1_rounds: int,
    stage2_rounds: int,
    sigma_mode: str,
    seed: int,
) -> Tuple[lgb.Booster, LightGBMLSS]:
    x = X_train.astype(float)
    y = np.asarray(y_train, dtype=float)

    p1 = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": int(seed),
    }
    d1 = lgb.Dataset(x, label=y, params={"verbose": -1})
    stage1 = lgb.train(p1, d1, num_boost_round=stage1_rounds, callbacks=[])

    mu_train = stage1.predict(x)
    init_score = np.column_stack([mu_train, np.zeros_like(mu_train)]).ravel(order="F")

    p2 = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 0.1,
        "feature_pre_filter": False,
        "force_col_wise": True,
        "verbosity": -1,
        "random_state": int(seed),
    }
    d2 = lgb.Dataset(x, label=y, init_score=init_score, params={"verbose": -1}, free_raw_data=False)
    stage2 = LightGBMLSS(distribution_for_mode(sigma_mode))
    stage2.start_values = np.array([float(np.mean(mu_train)), 0.0], dtype=np.float32)
    stage2.train(p2, d2, num_boost_round=stage2_rounds)

    return stage1, stage2


def _extract_sigma(dist_params: np.ndarray, n_rows: int) -> np.ndarray:
    params = dist_params.values if hasattr(dist_params, "values") else np.asarray(dist_params)
    if params.ndim == 1:
        if params.shape[0] == n_rows:
            sigma = params
        elif params.shape[0] == 2:
            sigma = np.repeat(params[-1], n_rows)
        else:
            sigma = np.repeat(float(params[-1]), n_rows)
    else:
        sigma = params[:, -1] if params.shape[1] > 1 else params[:, 0]
    return np.maximum(np.asarray(sigma, dtype=float), 1e-6)


def predict_quantiles(
    stage1: lgb.Booster,
    stage2: LightGBMLSS,
    test_x: pd.DataFrame,
    quantiles: np.ndarray,
    target_mode: str,
    current_obs: np.ndarray,
) -> np.ndarray:
    x = test_x.astype(float)
    mu = stage1.predict(x)
    dist_params = stage2.predict(x, pred_type="parameters")
    sigma = _extract_sigma(dist_params, len(x))

    q_model = norm.ppf(quantiles[None, :], loc=mu[:, None], scale=sigma[:, None])
    if target_mode == "level":
        q_log = q_model
    elif target_mode == "delta_log":
        base_log = np.log1p(np.clip(np.asarray(current_obs, dtype=float), 0.0, None))
        q_log = q_model + base_log[:, None]
    else:
        raise ValueError(f"Unknown target_mode={target_mode}")

    q_lin = np.expm1(q_log)
    return np.maximum(q_lin, 0.0)


def resolve_origin_date(anchor: pd.Timestamp, forecasting_weeks_file: Path) -> pd.Timestamp:
    if forecasting_weeks_file.exists():
        fw = pd.read_csv(forecasting_weeks_file)
        needed = {"origin_date", "horizon", "target_end_date"}
        if needed.issubset(fw.columns):
            fw["origin_date"] = pd.to_datetime(fw["origin_date"], errors="coerce")
            fw["target_end_date"] = pd.to_datetime(fw["target_end_date"], errors="coerce")
            # Anchor is the latest observed truth week, which corresponds to horizon 0.
            match = fw[(fw["horizon"] == 0) & (fw["target_end_date"] == anchor)]
            if len(match):
                return pd.to_datetime(match["origin_date"].max())

    # Fallback: next Wednesday from anchor date.
    days_to_wed = (2 - anchor.weekday()) % 7
    return anchor + pd.Timedelta(days=days_to_wed)


def run_prospective(cfg: RuntimeConfig) -> pd.DataFrame:
    loc_df = pd.read_csv(cfg.locations_file)
    if "iso2_code" not in loc_df.columns:
        raise ValueError(f"{cfg.locations_file} missing iso2_code column")
    locations = sorted(loc_df["iso2_code"].astype(str).unique().tolist())
    if cfg.locations_subset is not None:
        allowed = sorted(set(cfg.locations_subset))
        unknown = sorted(set(allowed).difference(set(locations)))
        if unknown:
            raise ValueError(f"Unknown locations in --locations scope: {unknown}")
        locations = allowed

    other_target = "ARI incidence" if cfg.target == "ILI incidence" else "ILI incidence"

    # Load full selected scope first, then keep only locations with recent truth.
    if cfg.anchor_date is not None:
        anchor = pd.to_datetime(cfg.anchor_date)
        target_df_all = _load_target_panel(
            cfg.data_file,
            cfg.target,
            locations,
            cutoff_date=anchor,
            calendar_end_date=anchor,
        )
    else:
        target_df_all = _load_target_panel(cfg.data_file, cfg.target, locations)
        anchor = pd.to_datetime(target_df_all[target_df_all["y"].notna()]["date"].max())
    recency_cutoff = anchor - pd.Timedelta(weeks=int(cfg.recent_weeks_required))

    last_obs = (
        target_df_all[target_df_all["y"].notna()]
        .groupby("location", as_index=False)["date"]
        .max()
        .rename(columns={"date": "last_obs_date"})
    )
    eligible_locations = sorted(
        last_obs.loc[last_obs["last_obs_date"] >= recency_cutoff, "location"].astype(str).tolist()
    )
    excluded_locations = sorted(set(locations).difference(set(eligible_locations)))
    print(
        f"[{cfg.target}] recent-truth filter: "
        f"require >= {recency_cutoff.date().isoformat()} "
        f"({cfg.recent_weeks_required} weeks), "
        f"eligible={len(eligible_locations)}, excluded={len(excluded_locations)}"
    )
    if excluded_locations:
        print(f"[{cfg.target}] excluded_locations={excluded_locations}")
    if not eligible_locations:
        raise RuntimeError(
            f"[{cfg.target}] no locations have truth in the last {cfg.recent_weeks_required} weeks"
        )

    target_df = _load_target_panel(
        cfg.data_file,
        cfg.target,
        eligible_locations,
        cutoff_date=anchor,
        calendar_end_date=anchor,
    )
    other_df = _load_target_panel(
        cfg.data_file,
        other_target,
        eligible_locations,
        cutoff_date=anchor,
        calendar_end_date=anchor,
    )
    if cfg.origin_date is not None:
        origin_date = pd.to_datetime(cfg.origin_date)
    else:
        origin_date = resolve_origin_date(anchor, cfg.forecasting_weeks_file)

    tgt_pivot = _pivot(target_df)
    oth_pivot = _pivot(other_df)

    donor_same = compute_top_donors(
        target_pivot=tgt_pivot,
        candidate_pivot=tgt_pivot,
        top_k=cfg.donor_top_k,
        min_overlap=cfg.min_overlap,
        target_mode=cfg.target_mode,
    )
    donor_other = compute_top_donors(
        target_pivot=tgt_pivot,
        candidate_pivot=oth_pivot,
        top_k=cfg.other_top_k,
        min_overlap=cfg.min_overlap,
        target_mode=cfg.target_mode,
    )

    feat = build_features(
        target_df=target_df,
        other_df=other_df,
        own_lags=cfg.own_lags,
        donor_lags=cfg.donor_lags,
        same_target_donors=donor_same,
        other_target_donors=donor_other,
        donor_top_k=cfg.donor_top_k,
        other_top_k=cfg.other_top_k,
    )

    pooled = build_pooled_examples(feat, cfg.max_horizons)
    feat_cols = feature_columns(pooled)

    train_df = pooled[(pooled["target_end_date"] <= anchor) & (pooled["date"] <= anchor)].copy()
    test_df = pooled[(pooled["date"] == anchor) & (pooled["horizon"] <= cfg.max_horizons)].copy()

    if test_df.empty:
        raise ValueError("No prospective rows found for anchor date")

    # Keep sparse rows with missing covariates; LightGBM handles NaNs natively.
    # Only target must be observed for training rows.
    train_df = train_df.dropna(subset=["target"]).copy()
    test_df_clean = test_df.copy()

    train_target = np.log1p(np.clip(train_df["target"].to_numpy(dtype=float), 0.0, None))
    if cfg.target_mode == "delta_log":
        base_log = np.log1p(np.clip(train_df["y_base"].to_numpy(dtype=float), 0.0, None))
        train_target = train_target - base_log
    valid = np.isfinite(train_target)
    train_df = train_df.loc[valid].reset_index(drop=True)
    train_target = train_target[valid]

    print(
        f"[{cfg.target}] anchor={anchor.date().isoformat()} "
        f"train_rows={len(train_df)} test_rows={len(test_df_clean)}"
    )

    bag_preds = []
    bag_locations: List[set[str]] = []
    if len(train_df) >= cfg.min_train_rows and not test_df_clean.empty:
        seasons = sorted(train_df["season"].dropna().unique())
        if seasons:
            bag_size = max(1, int(round(len(seasons) * cfg.bag_frac)))
            train_locations = sorted(train_df["location"].astype(str).unique().tolist())
            if not train_locations:
                raise RuntimeError(f"[{cfg.target}] no train locations available for bagging")
            if cfg.location_bag_frac <= 0:
                raise ValueError("location_bag_frac must be > 0")
            if cfg.location_bag_min < 1:
                raise ValueError("location_bag_min must be >= 1")
            loc_bag_size = max(int(np.ceil(len(train_locations) * cfg.location_bag_frac)), int(cfg.location_bag_min))
            loc_bag_size = min(max(1, loc_bag_size), len(train_locations))
            rng_seed = int(cfg.seed + int(anchor.value // 10**9))
            rng = np.random.default_rng(rng_seed)
            print(
                f"[{cfg.target}] season_bag_size={bag_size}/{len(seasons)} "
                f"location_bag_size={loc_bag_size}/{len(train_locations)}"
            )

            for b in range(cfg.num_bags):
                sampled = rng.choice(seasons, size=bag_size, replace=False)
                if loc_bag_size >= len(train_locations):
                    sampled_locs = train_locations
                else:
                    sampled_locs = sorted(rng.choice(train_locations, size=loc_bag_size, replace=False).tolist())
                season_mask = train_df["season"].isin(sampled).to_numpy()
                loc_mask = train_df["location"].isin(sampled_locs).to_numpy()
                bag_mask = season_mask & loc_mask
                bag_train = train_df.loc[bag_mask]
                bag_target = train_target[bag_mask]
                if len(bag_train) < cfg.min_train_rows:
                    continue
                try:
                    stage1, stage2 = fit_two_stage_one_bag(
                        X_train=bag_train.loc[:, feat_cols],
                        y_train=bag_target,
                        stage1_rounds=cfg.stage1_rounds,
                        stage2_rounds=cfg.stage2_rounds,
                        sigma_mode=cfg.sigma_mode,
                        seed=rng_seed + b,
                    )
                    q = predict_quantiles(
                        stage1,
                        stage2,
                        test_df_clean.loc[:, feat_cols],
                        QUANTILES,
                        target_mode=cfg.target_mode,
                        current_obs=test_df_clean["y_base"].to_numpy(dtype=float),
                    )
                    bag_preds.append(q)
                    bag_locations.append(set(sampled_locs))
                except Exception:
                    continue

    print(f"[{cfg.target}] successful_bags={len(bag_preds)}")

    if not bag_preds:
        raise RuntimeError(
            f"[{cfg.target}] model fitting produced zero successful bags; "
            "aborting instead of falling back."
        )

    rows = []
    modeled_pairs = set()

    bag_tensor = np.stack(bag_preds, axis=0)  # [n_bags, n_test_rows, n_quantiles]
    test_locs = test_df_clean["location"].astype(str).to_numpy()
    train_locs_valid = set(train_df["location"].astype(str).unique().tolist())
    loc_to_bag_idx: Dict[str, np.ndarray] = {}
    cold_start_locs: List[str] = []
    for loc in sorted(set(test_locs.tolist())):
        if loc not in train_locs_valid:
            # Location has no valid delta-training rows; use all successful bags.
            idx = list(range(len(bag_preds)))
            cold_start_locs.append(loc)
        else:
            idx = [i for i, loc_set in enumerate(bag_locations) if loc in loc_set]
            if not idx:
                raise RuntimeError(
                    f"[{cfg.target}] no successful bags included location={loc}; "
                    "increase num_bags or location_bag_frac/location_bag_min."
                )
        loc_to_bag_idx[loc] = np.asarray(idx, dtype=int)

    if cold_start_locs:
        print(
            f"[{cfg.target}] cold-start locations (no valid train rows, using all bags): "
            f"{sorted(cold_start_locs)}"
        )

    loc_cover = np.array([len(loc_to_bag_idx[l]) for l in sorted(loc_to_bag_idx)], dtype=int)
    print(
        f"[{cfg.target}] per-location bag coverage: min={int(loc_cover.min())} "
        f"median={float(np.median(loc_cover)):.1f} max={int(loc_cover.max())}"
    )

    for i, row in test_df_clean.reset_index(drop=True).iterrows():
        loc = row["location"]
        h = int(row["horizon"])
        te = pd.to_datetime(row["target_end_date"])
        bag_idx = loc_to_bag_idx[str(loc)]
        agg_q = np.median(bag_tensor[bag_idx, i, :], axis=0)
        modeled_pairs.add((loc, h))
        for q_idx, q_level in enumerate(QUANTILES):
            rows.append(
                {
                    "origin_date": origin_date.date().isoformat(),
                    "target": cfg.target,
                    "target_end_date": te.date().isoformat(),
                    "horizon": h,
                    "location": loc,
                    "output_type": "quantile",
                    "output_type_id": float(q_level),
                    "value": float(max(0.0, agg_q[q_idx])),
                }
            )

    pred_df = pd.DataFrame.from_records(rows)

    expected_pairs = {(r.location, int(r.horizon)) for r in test_df.itertuples(index=False)}
    missing_pairs = expected_pairs.difference(modeled_pairs)
    if missing_pairs:
        raise RuntimeError(
            f"[{cfg.target}] missing predictions for {len(missing_pairs)} "
            "location/horizon pairs; aborting."
        )

    pred_df = pred_df.sort_values(["location", "horizon", "output_type_id"]).reset_index(drop=True)
    return pred_df


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(df)} rows)")


def build_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        data_file=Path(args.data_file).resolve(),
        target=args.target,
        output=Path(args.output).resolve(),
        locations_file=Path(args.locations_file).resolve(),
        forecasting_weeks_file=Path(args.forecasting_weeks_file).resolve(),
        max_horizons=args.max_horizons,
        num_bags=args.num_bags,
        bag_frac=args.bag_frac,
        location_bag_frac=args.location_bag_frac,
        location_bag_min=args.location_bag_min,
        seed=args.seed,
        stage1_rounds=args.stage1_rounds,
        stage2_rounds=args.stage2_rounds,
        own_lags=parse_lag_string(args.own_lags),
        donor_lags=parse_lag_string(args.donor_lags),
        donor_top_k=args.donor_top_k,
        other_top_k=args.other_top_k,
        min_overlap=args.min_overlap,
        min_train_rows=args.min_train_rows,
        target_mode=args.target_mode,
        sigma_mode=args.sigma_mode,
        recent_weeks_required=args.recent_weeks_required,
        anchor_date=pd.to_datetime(args.anchor_date) if args.anchor_date else None,
        origin_date=pd.to_datetime(args.origin_date) if args.origin_date else None,
    )


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prospective joint two-stage model for RespiCast ILI/ARI")
    p.add_argument("--data-file", required=True)
    p.add_argument("--target", required=True, choices=["ILI incidence", "ARI incidence"])
    p.add_argument("--output", required=True)
    p.add_argument("--locations-file", required=True)
    p.add_argument("--forecasting-weeks-file", required=True)

    p.add_argument("--max-horizons", type=int, default=4)
    p.add_argument("--num-bags", type=int, default=80)
    p.add_argument("--bag-frac", type=float, default=0.7)
    p.add_argument(
        "--location-bag-frac",
        type=float,
        default=1.0,
        help="Fraction of locations sampled per bag (1.0 disables location subset bagging)",
    )
    p.add_argument(
        "--location-bag-min",
        type=int,
        default=1,
        help="Minimum number of locations retained in each bag",
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--stage1-rounds", type=int, default=200)
    p.add_argument("--stage2-rounds", type=int, default=150)
    p.add_argument("--own-lags", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,26,52")
    p.add_argument("--donor-lags", type=str, default="1,2,3,4,8,12")
    p.add_argument("--donor-top-k", type=int, default=4)
    p.add_argument("--other-top-k", type=int, default=2)
    p.add_argument("--min-overlap", type=int, default=30)
    p.add_argument("--min-train-rows", type=int, default=800)
    p.add_argument("--target-mode", choices=["level", "delta_log"], default="delta_log")
    p.add_argument("--sigma-mode", choices=["bounded", "unbounded"], default="bounded")
    p.add_argument(
        "--recent-weeks-required",
        type=int,
        default=4,
        help="Only forecast locations with at least one observed truth in the last N weeks",
    )
    p.add_argument("--anchor-date", default=None, help="Optional fixed anchor date (YYYY-MM-DD)")
    p.add_argument("--origin-date", default=None, help="Optional fixed origin date (YYYY-MM-DD)")

    ## New google data
    p.add_argument(
    "--google-trends-file",
    default=None,
    help="Path to preprocessed Google Trends CSV (optional)")
    return p


def main() -> None:
    args = make_parser().parse_args()
    cfg = build_config(args)
    pred = run_prospective(cfg)
    write_output(pred, cfg.output)


if __name__ == "__main__":
    main()
