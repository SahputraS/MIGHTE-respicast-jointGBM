#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from build_long_timeseries import resolve_long_timeseries
from model_joint_twostage_eu import RuntimeConfig, run_prospective


def target_slug(target: str) -> str:
    if target == "ILI incidence":
        return "ILI"
    if target == "ARI incidence":
        return "ARI"
    return target.replace(" ", "_")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prospective RespiCast forecasts (ILI and/or ARI)")
    parser.add_argument("--hub-dir", default="RespiCast-SyndromicIndicators")
    parser.add_argument("--targets", default="ILI,ARI", help="Comma-separated: ILI,ARI")
    parser.add_argument("--model-id", default="MIGHTE-jointGBM")
    parser.add_argument(
        "--locations",
        default=None,
        help="Optional comma-separated ISO2 locations to forecast (default: all hub locations)",
    )

    parser.add_argument("--canonical-data", default="data/processed/respicast_long_latest.csv")
    parser.add_argument("--summary-json", default="data/processed/respicast_long_summary.json")

    parser.add_argument("--raw-dir", default="forecasts/prospective/raw")
    parser.add_argument("--submission-dir", default="forecasts/prospective/submission")
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
        help="Only forecast locations with at least one observed truth in last N weeks",
    )

    # Google data
    parser.add_argument(
        "--google-trends-file",
        default=None,
        help="Path to preprocessed Google Trends CSV (optional)",
    )
    # -------- I add this bloc for gridsearch and exclude covid ---------
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--lambda-l2", type=float, default=0.1)

    parser.add_argument("--s2-num-leaves", type=int, default=None)         # I ADD
    parser.add_argument("--s2-learning-rate", type=float, default=None)    # I ADD
    parser.add_argument("--s2-min-child-samples", type=int, default=None)  # I ADD
    parser.add_argument("--s2-feature-fraction", type=float, default=None) # I ADD
    parser.add_argument("--s2-max-depth", type=int, default=6)             # I ADD
    
    parser.add_argument("--exclude-covid", action="store_true",
                        help="Exclude COVID period (2019-10 to 2022-09) from training")
    # ---------------------------------------------------

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

    locations_file = hub_dir / "supporting-files" / "locations_iso2_codes.csv"
    forecasting_weeks = hub_dir / "supporting-files" / "forecasting_weeks.csv"

    sub_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    targets = parse_targets(args.targets)
    location_scope = parse_locations(args.locations)
    pred_by_target = {}
    origin_by_target = {}

    for target in targets:
        cfg = RuntimeConfig(
            data_file=canonical_path,
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
            stage1_rounds=args.stage1_rounds,
            stage2_rounds=args.stage2_rounds,
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
            google_trends_file=Path(args.google_trends_file) if args.google_trends_file else None,  # I ADD
            num_leaves=args.num_leaves,                    # I ADD
            learning_rate=args.learning_rate,              # I ADD
            min_child_samples=args.min_child_samples,      # I ADD
            feature_fraction=args.feature_fraction,        # I ADD
            lambda_l2=args.lambda_l2,                      # I ADD
            s2_num_leaves=args.s2_num_leaves,              # I ADD
            s2_learning_rate=args.s2_learning_rate,        # I ADD
            s2_min_child_samples=args.s2_min_child_samples,# I ADD
            s2_feature_fraction=args.s2_feature_fraction,  # I ADD
            s2_max_depth=args.s2_max_depth,                # I ADD
            exclude_covid=args.exclude_covid,              # I ADD
        )

        pred = run_prospective(cfg)
        if pred.empty:
            raise ValueError(f"No prospective forecasts generated for {target}")

        origin_date = str(pred["origin_date"].iloc[0])
        pred_by_target[target] = pred
        origin_by_target[target] = origin_date

        if args.save_raw:
            slug = target_slug(target)
            raw_path = raw_dir / f"{origin_date}-{args.model_id}-{slug}-raw.csv"
            pred.to_csv(raw_path, index=False)
            print(f"Saved {target} raw: {raw_path} ({len(pred)} rows)")

    origin_dates = sorted(set(origin_by_target.values()))
    if len(origin_dates) != 1:
        raise RuntimeError(
            f"Targets produced inconsistent reference dates: {origin_by_target}. "
            "Aborting to avoid mixed-date submission."
        )

    origin_date = origin_dates[0]
    combined = pd.concat([pred_by_target[t] for t in targets], ignore_index=True)
    combined = combined.sort_values(
        ["target", "location", "horizon", "output_type_id"]
    ).reset_index(drop=True)

    sub_path = sub_dir / f"{origin_date}-{args.model_id}.csv"
    combined.to_csv(sub_path, index=False)
    print(f"Saved combined submission: {sub_path} ({len(combined)} rows)")

    # Remove legacy same-date files from older naming schemes to avoid ambiguity in viz.
    legacy_patterns = [
        f"{origin_date}-{args.model_id}-*.csv",
        f"{origin_date}-{args.model_id}_*.csv",
    ]
    removed = 0
    for patt in legacy_patterns:
        for p in sub_dir.glob(patt):
            if p.name == sub_path.name:
                continue
            p.unlink(missing_ok=True)
            removed += 1
    if removed:
        print(f"Removed {removed} legacy same-date submission file(s)")


if __name__ == "__main__":
    main()
