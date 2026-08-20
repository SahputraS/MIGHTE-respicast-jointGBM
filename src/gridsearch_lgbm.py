#!/usr/bin/env python
"""Gridsearch for the ILI/ARI joint two-stage LightGBM model.

Consolidates the duplicated stage-1 (RMSE) + stage-2 (WIS) Optuna search
previously copy-pasted between 26_27_Gridsearch_and_Run_LightGBM_ILI.ipynb
and 26_27_Gridsearch_and_Run_LightGBM_ARI.ipynb into one function, callable
for either target via `run_gridsearch(target="ILI" | "ARI", ...)`.

Runs against the preprocessed (denoised+detrended) Google Trends features
only:
1. builds the pooled feature matrix,
2. runs a stage-1 Optuna search minimizing validation RMSE (with early
   stopping to pick boosting rounds),
3. runs a stage-2 Optuna search minimizing validation WIS, frozen on the
   stage-1 rounds,
4. packs both into one params dict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb

from model_joint_twostage_eu import (
    _load_target_panel, _pivot, compute_top_donors,
    build_features, build_pooled_examples, feature_columns,
    fit_two_stage_one_bag, predict_quantiles, QUANTILES,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGET_COLUMN = {"ILI": "ILI incidence", "ARI": "ARI incidence"}
GT_VARIANT = "gt_proc"


def build_matrix(
    data_file,
    target_col: str,
    other_col: str,
    locations: Sequence[str],
    anchor: pd.Timestamp,
    exclude_covid: bool,
    own_lags: Sequence[int],
    donor_lags: Sequence[int],
    donor_top_k: int,
    other_top_k: int,
    gt_file: Optional[str] = None,
    include_gt_lead: bool = True,
    gt_min_corr: float = 0.3,
) -> Tuple[pd.DataFrame, np.ndarray, list]:
    target_df = _load_target_panel(data_file, target_col, locations,
                                    cutoff_date=anchor, calendar_end_date=anchor,
                                    exclude_covid=exclude_covid)
    other_df = _load_target_panel(data_file, other_col, locations,
                                   cutoff_date=anchor, calendar_end_date=anchor,
                                   exclude_covid=exclude_covid)

    tgt_pivot, oth_pivot = _pivot(target_df), _pivot(other_df)
    donor_same = compute_top_donors(tgt_pivot, tgt_pivot, donor_top_k, 30, "delta_log")
    donor_other = compute_top_donors(tgt_pivot, oth_pivot, other_top_k, 30, "delta_log")

    gt_df = pd.read_csv(gt_file, parse_dates=["date"]) if gt_file is not None else None

    feat = build_features(
        target_df, other_df,
        own_lags=list(own_lags), donor_lags=list(donor_lags),
        same_target_donors=donor_same, other_target_donors=donor_other,
        donor_top_k=donor_top_k, other_top_k=other_top_k, gt_df=gt_df,
        include_gt_lead=include_gt_lead, gt_min_corr=gt_min_corr,
    )
    pooled = build_pooled_examples(feat, 4)
    feat_cols = feature_columns(pooled)

    df = pooled.dropna(subset=["target"]).copy()
    y_log = np.log1p(np.clip(df["target"].to_numpy(float), 0, None))
    base = np.log1p(np.clip(df["y_base"].to_numpy(float), 0, None))
    y = y_log - base
    ok = np.isfinite(y)
    df, y = df.loc[ok].reset_index(drop=True), y[ok]
    return df, y, feat_cols


def l2_for_config(df, y, feat_cols, params, seed, cutoff_q, num_threads: int = 0) -> Tuple[float, int]:
    """Native delta-log L2 (MSE) -- self-consistent: early stopping and the returned
    score are the same metric, in the same space the model actually predicts in.
    Chosen over cases-space RMSE because the log-space compression keeps every
    location roughly scale-balanced; raw cases-space RMSE is dominated by whichever
    high-volume locations/weeks have the largest absolute case counts, and (being
    noisier round-to-round) can reward a shallow, undertrained model that got a lucky
    validation read over one that actually generalizes."""
    cut = df["date"].quantile(cutoff_q)
    tr = (df["date"] <= cut).to_numpy()
    X_tr, y_tr = df.loc[tr, feat_cols].astype(float), y[tr]
    X_va = df.loc[~tr, feat_cols].astype(float)
    y_va = y[~tr]

    p1 = {
        "objective": "regression", "metric": "l2",
        "learning_rate": params["learning_rate"],
        "num_leaves": params["num_leaves"],
        "min_child_samples": params["min_child_samples"],
        "feature_fraction": params["feature_fraction"],
        "bagging_fraction": 0.9, "bagging_freq": 1,
        "seed": seed, "deterministic": True, "force_row_wise": True,
        "verbosity": -1,
        "num_threads": num_threads,
    }
    d_tr = lgb.Dataset(X_tr, label=y_tr)
    d_va = lgb.Dataset(X_va, label=y_va, reference=d_tr)
    model = lgb.train(
        p1, d_tr,
        num_boost_round=3000,  # low learning rates genuinely need the room; early stopping is self-consistent now
        valid_sets=[d_va],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    l2 = float(model.best_score["valid_0"]["l2"])
    return l2, int(model.best_iteration)


def run_stage1_study(
    name, df, y, feat_cols, seed, n_trials, cutoff_q, verbose=True,
    num_threads: int = 0, n_jobs: int = 1,
) -> optuna.Study:
    """num_threads: threads per individual LightGBM fit (0 = LightGBM's "all cores"
    default). n_jobs: how many Optuna trials run concurrently (Optuna threading, not
    processes). When n_jobs > 1, pass a num_threads that keeps n_jobs * num_threads
    within your actual core budget -- otherwise the trials fight each other for
    cores instead of speeding things up."""
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 50, 130),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        }
        l2, best_round = l2_for_config(df, y, feat_cols, params, seed, cutoff_q, num_threads=num_threads)
        trial.set_user_attr("best_round", best_round)
        return l2

    if verbose:
        print(f"=== Stage 1 (delta-log L2) optimizing: {name} ===  (n_jobs={n_jobs}, num_threads={num_threads})")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose)
    return study


def wis_vectorized(pred_q: np.ndarray, truth: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    q = np.asarray(quantiles, float)
    o = np.argsort(q)
    q, pred_q = q[o], pred_q[:, o]
    median = pred_q[:, int(np.argmin(np.abs(q - 0.5)))]
    interval_sum = np.zeros(len(truth))
    k = 0
    for lp in q[q < 0.5]:
        li = int(np.argmin(np.abs(q - lp)))
        ui = int(np.argmin(np.abs(q - (1 - lp))))
        lq, uq = pred_q[:, li], pred_q[:, ui]
        alpha = 2 * lp
        is_alpha = ((uq - lq)
                    + (2 / alpha) * (lq - truth) * (truth < lq)
                    + (2 / alpha) * (truth - uq) * (truth > uq))
        interval_sum += (alpha / 2) * is_alpha
        k += 1
    return (0.5 * np.abs(truth - median) + interval_sum) / (k + 0.5)


def run_stage2_study(
    name, df, y, feat_cols, stage1_best, seed, n_trials, cutoff_q, verbose=True,
    num_threads: int = 0, n_jobs: int = 1,
) -> optuna.Study:
    """See run_stage1_study's docstring for num_threads/n_jobs semantics."""
    cut = df["date"].quantile(cutoff_q)
    tr = (df["date"] <= cut).to_numpy()
    va = ~tr
    X_tr, y_tr = df.loc[tr, feat_cols].astype(float), y[tr]
    X_va = df.loc[va, feat_cols].astype(float)
    base_va = df.loc[va, "y_base"].to_numpy(float)
    actual = df.loc[va, "target"].to_numpy(float)
    ok = np.isfinite(actual)

    def objective(trial):
        stage1, stage2 = fit_two_stage_one_bag(
            X_train=X_tr, y_train=y_tr,
            stage1_rounds=stage1_best["rounds"], seed=seed,
            num_leaves=stage1_best["num_leaves"], learning_rate=stage1_best["learning_rate"],
            min_child_samples=stage1_best["min_child_samples"], feature_fraction=stage1_best["feature_fraction"],
            stage2_rounds=trial.suggest_int("stage2_rounds", 30, 250),
            sigma_mode="bounded",
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
            s2_min_child_samples=trial.suggest_int("s2_min_child_samples", 10, 120),
            s2_num_leaves=None, s2_learning_rate=None, s2_feature_fraction=None, s2_max_depth=6,
            num_threads=num_threads,
        )
        pred_q = predict_quantiles(stage1, stage2, X_va, QUANTILES, target_mode="delta_log", current_obs=base_va)
        pred_q = np.clip(np.nan_to_num(pred_q, nan=0.0, posinf=1e7, neginf=0.0), 0.0, 1e7)
        score = float(np.mean(wis_vectorized(pred_q[ok], actual[ok], QUANTILES)))
        return score if np.isfinite(score) else 1e12

    if verbose:
        print(f"=== Stage 2 (WIS) optimizing: {name}  (stage1 rounds frozen at {stage1_best['rounds']}, "
              f"n_jobs={n_jobs}, num_threads={num_threads}) ===")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=verbose)
    study.set_user_attr("stage1_rounds", stage1_best["rounds"])
    return study


def pack(study_s1: optuna.Study, study_s2: optuna.Study) -> Dict:
    return {
        **study_s1.best_params,
        "rounds": study_s1.best_trial.user_attrs["best_round"],
        "stage1_rounds_for_s2": study_s2.user_attrs["stage1_rounds"],
        **{f"s2_{k}" if not k.startswith(("stage2", "lambda", "sigma", "s2")) else k: v
           for k, v in study_s2.best_params.items()},
    }


def run_gridsearch(
    target: str,
    data_file,
    locations_file,
    anchor,
    gt_file: str,
    exclude_covid: bool = True,
    include_gt_lead: bool = True,
    gt_min_corr: float = 0.3,
    own_lags: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 26, 52),
    donor_lags: Sequence[int] = (1, 2, 3, 4, 8, 12),
    donor_top_k: int = 4,
    other_top_k: int = 2,
    cutoff_q: float = 0.75,
    seed: int = 4321,
    n_trials_stage1: int = 70,
    n_trials_stage2: int = 60,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Run the gridsearch for one target, using the preprocessed Google Trends
    features (denoised+detrended) only.

    `target` is "ILI" or "ARI"; the other syndromic indicator is used
    automatically as the secondary/donor series. `gt_file` is the
    preprocessed Google Trends wide CSV and is required.

    Returns a dict with a single "gt_proc" key mapping to the combined
    stage-1 + stage-2 best params, in the same shape as
    forecast_backtest.py / forecast_prospective.py expect.
    """
    target = target.upper()
    if target not in TARGET_COLUMN:
        raise ValueError(f"target must be one of {list(TARGET_COLUMN)}, got {target!r}")
    if not gt_file:
        raise ValueError("gt_file (preprocessed Google Trends CSV) is required")
    other = "ARI" if target == "ILI" else "ILI"
    target_col, other_col = TARGET_COLUMN[target], TARGET_COLUMN[other]

    anchor = pd.to_datetime(anchor)
    loc_df = pd.read_csv(locations_file)
    locations = sorted(loc_df["iso2_code"].astype(str).unique().tolist())

    if verbose:
        print(f"Building feature matrix for {target} (secondary={other}, GT=preprocessed)...")
    df, y, cols = build_matrix(
        data_file, target_col, other_col, locations, anchor, exclude_covid,
        own_lags, donor_lags, donor_top_k, other_top_k, gt_file=gt_file,
        include_gt_lead=include_gt_lead, gt_min_corr=gt_min_corr,
    )
    if verbose:
        print(f"  {df.shape[0]} rows, {len(cols)} features")

    study_s1 = run_stage1_study(GT_VARIANT, df, y, cols, seed, n_trials_stage1, cutoff_q, verbose)
    stage1_best = {**study_s1.best_params, "rounds": study_s1.best_trial.user_attrs["best_round"]}
    study_s2 = run_stage2_study(GT_VARIANT, df, y, cols, stage1_best, seed, n_trials_stage2, cutoff_q, verbose)
    best = {GT_VARIANT: pack(study_s1, study_s2)}
    if verbose:
        print(f"best stage1 L2={study_s1.best_value:.5f}  best stage2 WIS={study_s2.best_value:.4f}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(best, f, indent=2)
        if verbose:
            print(f"Saved best params to {output_path}")

    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LightGBM gridsearch (preprocessed Google Trends only) for ILI or ARI"
    )
    parser.add_argument("--target", required=True, choices=["ILI", "ARI"])
    parser.add_argument("--data-file", default="data/processed/respicast_long_latest.csv")
    parser.add_argument("--hub-dir", default="RespiCast-SyndromicIndicators",
                         help="RespiCast-SyndromicIndicators checkout (for supporting-files/locations_iso2_codes.csv)")
    parser.add_argument("--anchor-date", default="2025-05-01")
    parser.add_argument("--gt-file", required=True, help="Path to denoised+detrended Google Trends wide CSV")
    parser.add_argument("--no-exclude-covid", dest="exclude_covid", action="store_false")
    parser.add_argument("--no-gt-lead", dest="include_gt_lead", action="store_false",
                         help="Disable the GT 'nowcast' lag-1 feature (on by default)")
    parser.add_argument("--gt-min-corr", type=float, default=0.3,
                         help="Drop a GT term/location pair whose same-week |correlation| with the target is below this")
    parser.set_defaults(exclude_covid=True, include_gt_lead=True)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--cutoff-q", type=float, default=0.75)
    parser.add_argument("--n-trials-stage1", type=int, default=70)
    parser.add_argument("--n-trials-stage2", type=int, default=60)
    parser.add_argument("--output", default=None,
                         help="Path to write best-params JSON (default: best_params_26_27_<target>.json)")
    args = parser.parse_args()

    locations_file = Path(args.hub_dir) / "supporting-files" / "locations_iso2_codes.csv"
    output_path = args.output or f"best_params_26_27_{args.target.lower()}.json"

    best = run_gridsearch(
        target=args.target,
        data_file=args.data_file,
        locations_file=locations_file,
        anchor=args.anchor_date,
        gt_file=args.gt_file,
        exclude_covid=args.exclude_covid,
        include_gt_lead=args.include_gt_lead,
        gt_min_corr=args.gt_min_corr,
        seed=args.seed,
        cutoff_q=args.cutoff_q,
        n_trials_stage1=args.n_trials_stage1,
        n_trials_stage2=args.n_trials_stage2,
        output_path=output_path,
    )
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
