#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from build_long_timeseries import resolve_long_timeseries, resolve_long_timeseries_asof
from model_joint_twostage_eu import RuntimeConfig, run_prospective


def target_slug(target: str) -> str:
    if target == "ILI incidence":
        return "ILI"
    if target == "ARI incidence":
        return "ARI"
    return target.replace(" ", "_")


def load_best_params(params_dir: Path, target: str) -> dict:
    """Load gridsearch-tuned hyperparameters for one target.

    Looks for the JSON src/gridsearch_lgbm.py (or Gridsearch_LightGBM.ipynb)
    saves (best_params_ablation_<target>.json, under the "gt_proc" key),
    falling back to the legacy best_params_ablation*.json names.
    """
    slug = target_slug(target)
    candidates = [
        params_dir / f"best_params_ablation_{slug.lower()}.json",
        params_dir / (f"best_params_ablation({slug.lower()}).json" if slug == "ARI" else "best_params_ablation.json"),
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            params = data.get("gt_proc", data)
            print(f"[{target}] loaded tuned params from {path.name}")
            return params
    raise FileNotFoundError(
        f"--params-dir given but no best-params JSON found for {target}. "
        f"Checked: {[p.name for p in candidates]}"
    )


def parse_targets(text: str) -> List[str]:
    tokens = [t.strip().upper() for t in text.split(",") if t.strip()]
    out = []
    for t in tokens:
        if t == "ILI":
            out.append("ILI incidence")
        elif t == "ARI":
            out.append("ARI incidence")
        else:
            raise ValueError(f"Unsupported target token: {t}")
    if not out:
        raise ValueError("At least one target must be selected")
    return out


def parse_locations(text: str | None) -> List[str] | None:
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    tokens = [tok.strip().upper() for tok in s.split(",") if tok.strip()]
    if not tokens:
        return None
    return sorted(set(tokens))


def choose_backtest_origins(
    forecasting_weeks_file: Path,
    max_truth_date: pd.Timestamp,
    start_origin_date: pd.Timestamp,
    end_origin_date: pd.Timestamp | None,
    max_origins: int | None,
) -> pd.DataFrame:
    fw = pd.read_csv(forecasting_weeks_file)
    need = {"origin_date", "horizon", "target_end_date"}
    missing = need.difference(fw.columns)
    if missing:
        raise ValueError(f"{forecasting_weeks_file} missing required columns: {sorted(missing)}")

    fw["origin_date"] = pd.to_datetime(fw["origin_date"], errors="coerce")
    fw["target_end_date"] = pd.to_datetime(fw["target_end_date"], errors="coerce")
    fw["horizon"] = pd.to_numeric(fw["horizon"], errors="coerce")

    fw0 = fw[fw["horizon"] == 0].copy()
    fw0 = fw0.dropna(subset=["origin_date", "target_end_date"])

    out = fw0[fw0["target_end_date"] <= max_truth_date].copy()
    out = out[out["origin_date"] >= start_origin_date].copy()
    if end_origin_date is not None:
        out = out[out["origin_date"] <= end_origin_date].copy()

    out = out.sort_values("origin_date").drop_duplicates(subset=["origin_date"], keep="last")
    if max_origins is not None and max_origins > 0:
        out = out.head(max_origins).copy()
    out = out.reset_index(drop=True)
    return out.loc[:, ["origin_date", "target_end_date"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling retrospective backtest for RespiCast ILI/ARI")
    parser.add_argument("--hub-dir", default="RespiCast-SyndromicIndicators")
    parser.add_argument("--targets", default="ILI,ARI", help="Comma-separated: ILI,ARI")
    parser.add_argument("--model-id", default="ISI_LightGBM")
    parser.add_argument(
        "--locations",
        default=None,
        help="Optional comma-separated ISO2 locations to forecast (default: all hub locations)",
    )
    parser.add_argument(
        "--start-origin-date",
        default="2025-10-01",
        help="First origin date to include in rolling backtest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-origin-date",
        default=None,
        help="Optional last origin date to include (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-origins",
        type=int,
        default=None,
        help="Optional cap on number of rolling origins (for testing)",
    )

    parser.add_argument("--canonical-data", default="data/processed/respicast_long_latest.csv")
    parser.add_argument("--summary-json", default="data/processed/respicast_long_summary.json")

    parser.add_argument("--raw-dir", default="forecasts/retrospective/raw")
    parser.add_argument("--submission-dir", default="forecasts/retrospective/submission")
    parser.add_argument("--save-raw", action="store_true", help="Also save duplicate raw forecast files")

    parser.add_argument("--max-horizons", type=int, default=4)
    parser.add_argument("--num-bags", type=int, default=80)
    parser.add_argument("--bag-frac", type=float, default=0.7)
    parser.add_argument(
        "--location-bag-frac",
        type=float,
        default=1.0,
        help="Fraction of locations sampled per bag (1.0 disables location subset bagging)",
    )
    parser.add_argument(
        "--location-bag-min",
        type=int,
        default=1,
        help="Minimum number of locations retained in each bag",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stage1-rounds", type=int, default=200)
    parser.add_argument("--stage2-rounds", type=int, default=150)
    parser.add_argument("--own-lags", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,26,52")
    parser.add_argument("--donor-lags", type=str, default="1,2,3,4,8,12")
    parser.add_argument("--donor-top-k", type=int, default=4)
    parser.add_argument("--other-top-k", type=int, default=2)
    parser.add_argument("--min-overlap", type=int, default=30)
    parser.add_argument("--min-train-rows", type=int, default=800)
    parser.add_argument("--target-mode", choices=["level", "delta_log"], default="delta_log")
    parser.add_argument("--sigma-mode", choices=["bounded", "unbounded"], default="bounded")
    parser.add_argument(
        "--recent-weeks-required",
        type=int,
        default=4,
        help="Only forecast locations with at least one observed truth in last N weeks as-of each origin",
    )

    parser.add_argument(
        "--google-trends-file",
        default=None,
        help="Path to preprocessed Google Trends CSV (optional)",
    ) # I add for Google Trend

    parser.add_argument("--num-leaves", type=int, default=31) # I add for gridsearch
    parser.add_argument("--learning-rate", type=float, default=0.05) # I add for gridsearch
    parser.add_argument("--min-child-samples", type=int, default=20) # I add for gridsearch
    parser.add_argument("--feature-fraction", type=float, default=0.9) # I add for gridsearch
    parser.add_argument("--lambda-l2", type=float, default=0.1) # I add for gridsearch

    parser.add_argument("--s2-num-leaves", type=int, default=None)         # I ADD
    parser.add_argument("--s2-learning-rate", type=float, default=None)    # I ADD
    parser.add_argument("--s2-min-child-samples", type=int, default=None)  # I ADD
    parser.add_argument("--s2-feature-fraction", type=float, default=None) # I ADD
    parser.add_argument("--s2-max-depth", type=int, default=6)             # I ADD
    
    parser.add_argument("--exclude-covid", action="store_true",
                        help="Exclude the COVID-disrupted period (2019-10 to 2022-09) from training") # I add for exclude COVID period
    parser.add_argument("--no-gt-lead", dest="include_gt_lead", action="store_false",
                        help="Disable the GT 'nowcast' lag-1 feature (on by default; no-op without --google-trends-file)")
    parser.add_argument("--gt-min-corr", type=float, default=0.3,
                        help="Drop a GT term/location pair whose same-week |correlation| with the target is below this")
    parser.set_defaults(include_gt_lead=True)
    parser.add_argument("--num-threads", type=int, default=0,
                        help="Threads per LightGBM fit (0 = LightGBM's own 'use all available cores' default). "
                             "Set this on a shared/quota-limited server -- leaving it at 0 there can crash with "
                             "'libgomp: Thread creation failed: Resource temporarily unavailable' if the machine's "
                             "real CPU quota is far below its reported core count.")
    parser.add_argument(
        "--params-dir",
        default=None,
        help=(
            "Directory containing gridsearch best-params JSON (best_params_ablation_<target>.json, "
            "falling back to legacy best_params_ablation*.json names). When set, overrides "
            "--num-leaves/--learning-rate/--min-child-samples/--feature-fraction/--stage1-rounds/"
            "--stage2-rounds/--lambda-l2/--s2-min-child-samples with each target's own tuned values, "
            "loaded once and reused for every backtest origin."
        ),
    )
    parser.add_argument(
        "--snapshots-only",
        action="store_true",
        help=(
            "Vintage-aware backtest: at each origin, rebuild the canonical series from only "
            "dated snapshot files available by that origin (excludes latest-*.csv, which always "
            "reflects 'now'). Avoids the data-revision look-ahead bias where a backtest trains "
            "on truth values that have since been revised/backfilled past what a real forecaster "
            "had at that origin. Off by default (uses the single current canonical for every "
            "origin, like a normal backtest); slower, since the canonical series is rebuilt once "
            "per origin instead of once total."
        ),
    )

    args = parser.parse_args()

    hub_dir = Path(args.hub_dir).resolve()
    canonical_path = Path(args.canonical_data).resolve()
    summary_path = Path(args.summary_json).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    sub_dir = Path(args.submission_dir).resolve()

    canonical = resolve_long_timeseries(hub_dir)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(canonical_path, index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        canonical.groupby("target").size().rename("rows").to_json(indent=2),
        encoding="utf-8",
    )
    print(f"Saved canonical long series: {canonical_path} ({len(canonical)} rows)")

    if args.snapshots_only:
        vintage_dir = canonical_path.parent / "vintage"
        vintage_dir.mkdir(parents=True, exist_ok=True)
        print(f"--snapshots-only: rebuilding a vintage-limited canonical series per origin under {vintage_dir}")

    locations_file = hub_dir / "supporting-files" / "locations_iso2_codes.csv"
    forecasting_weeks = hub_dir / "supporting-files" / "forecasting_weeks.csv"

    sub_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    targets = parse_targets(args.targets)
    location_scope = parse_locations(args.locations)

    if args.params_dir:
        params_by_target = {t: load_best_params(Path(args.params_dir).resolve(), t) for t in targets}
    else:
        params_by_target = None

    max_truth_date = pd.to_datetime(canonical["truth_date"], errors="coerce").max()
    start_origin = pd.to_datetime(args.start_origin_date)
    end_origin = pd.to_datetime(args.end_origin_date) if args.end_origin_date else None

    rolling_dates = choose_backtest_origins(
        forecasting_weeks_file=forecasting_weeks,
        max_truth_date=max_truth_date,
        start_origin_date=start_origin,
        end_origin_date=end_origin,
        max_origins=args.max_origins,
    )
    if rolling_dates.empty:
        raise RuntimeError(
            "No eligible backtest origin dates found after filtering by start/end and available truth."
        )

    print(
        f"Backtest origins: {len(rolling_dates)} from "
        f"{rolling_dates['origin_date'].min().date().isoformat()} to "
        f"{rolling_dates['origin_date'].max().date().isoformat()}"
    )

    for i, row in enumerate(rolling_dates.itertuples(index=False), start=1):
        origin_date = pd.to_datetime(row.origin_date)
        anchor_date = pd.to_datetime(row.target_end_date)
        print(
            f"[{i}/{len(rolling_dates)}] origin={origin_date.date().isoformat()} "
            f"anchor={anchor_date.date().isoformat()}"
        )

        if args.snapshots_only:
            vintage = resolve_long_timeseries_asof(hub_dir, origin_date)
            origin_data_file = vintage_dir / f"{origin_date.date().isoformat()}-respicast_long.csv"
            vintage.to_csv(origin_data_file, index=False)
            print(f"  vintage: {len(vintage)} rows, max truth_date={vintage['truth_date'].max()}")
        else:
            origin_data_file = canonical_path

        pred_by_target = {}
        for target in targets:
            if params_by_target is not None:
                params = params_by_target[target]
                stage1_rounds = params["rounds"]
                stage2_rounds = params["stage2_rounds"]
                num_leaves = params["num_leaves"]
                learning_rate = params["learning_rate"]
                min_child_samples = params["min_child_samples"]
                feature_fraction = params["feature_fraction"]
                lambda_l2 = params["lambda_l2"]
                s2_min_child_samples = params["s2_min_child_samples"]
            else:
                stage1_rounds = args.stage1_rounds
                stage2_rounds = args.stage2_rounds
                num_leaves = args.num_leaves
                learning_rate = args.learning_rate
                min_child_samples = args.min_child_samples
                feature_fraction = args.feature_fraction
                lambda_l2 = args.lambda_l2
                s2_min_child_samples = args.s2_min_child_samples

            cfg = RuntimeConfig(
                data_file=origin_data_file,
                target=target,
                output=sub_dir / f"tmp_{target_slug(target)}.csv",
                locations_file=locations_file,
                forecasting_weeks_file=forecasting_weeks,
                max_horizons=args.max_horizons,
                num_bags=args.num_bags,
                bag_frac=args.bag_frac,
                location_bag_frac=args.location_bag_frac,
                location_bag_min=args.location_bag_min,
                seed=args.seed,
                stage1_rounds=stage1_rounds,
                stage2_rounds=stage2_rounds,
                own_lags=[int(x.strip()) for x in args.own_lags.split(",") if x.strip()],
                donor_lags=[int(x.strip()) for x in args.donor_lags.split(",") if x.strip()],
                donor_top_k=args.donor_top_k,
                other_top_k=args.other_top_k,
                min_overlap=args.min_overlap,
                min_train_rows=args.min_train_rows,
                target_mode=args.target_mode,
                sigma_mode=args.sigma_mode,
                locations_subset=location_scope,
                recent_weeks_required=args.recent_weeks_required,
                anchor_date=anchor_date,
                origin_date=origin_date,
                google_trends_file=Path(args.google_trends_file) if args.google_trends_file else None, # I ADD
                num_leaves=num_leaves,                          # I ADD
                learning_rate=learning_rate,                    # I ADD
                min_child_samples=min_child_samples,            # I ADD
                feature_fraction=feature_fraction,              # I ADD
                lambda_l2=lambda_l2,                            # I ADD
                s2_num_leaves=args.s2_num_leaves,              # I ADD
                s2_learning_rate=args.s2_learning_rate,        # I ADD
                s2_min_child_samples=s2_min_child_samples,      # I ADD
                s2_feature_fraction=args.s2_feature_fraction,  # I ADD
                s2_max_depth=args.s2_max_depth,                # I ADD

                exclude_covid=args.exclude_covid,                  # I ADD
                include_gt_lead=args.include_gt_lead,              # I ADD
                gt_min_corr=args.gt_min_corr,                      # I ADD
                num_threads=args.num_threads,                      # I ADD
            )
            pred = run_prospective(cfg)
            if pred.empty:
                raise RuntimeError(
                    f"No retrospective forecasts generated for {target} at {origin_date.date().isoformat()}"
                )
            pred_by_target[target] = pred

            if args.save_raw:
                slug = target_slug(target)
                raw_path = raw_dir / f"{origin_date.date().isoformat()}-{args.model_id}-{slug}-raw.csv"
                pred.to_csv(raw_path, index=False)

        combined = pd.concat([pred_by_target[t] for t in targets], ignore_index=True)
        combined = combined.sort_values(
            ["target", "location", "horizon", "output_type_id"]
        ).reset_index(drop=True)

        out_name = f"{origin_date.date().isoformat()}-{args.model_id}.csv"
        out_path = sub_dir / out_name
        combined.to_csv(out_path, index=False)

        # Remove legacy same-date files from older naming schemes.
        legacy_patterns = [
            f"{origin_date.date().isoformat()}-{args.model_id}-*.csv",
            f"{origin_date.date().isoformat()}-{args.model_id}_*.csv",
        ]
        for patt in legacy_patterns:
            for p in sub_dir.glob(patt):
                if p.name == out_path.name:
                    continue
                p.unlink(missing_ok=True)

    print(f"Backtest forecasts saved under: {sub_dir}")


if __name__ == "__main__":
    main()
