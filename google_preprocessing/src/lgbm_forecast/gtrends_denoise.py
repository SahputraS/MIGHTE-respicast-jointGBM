"""Rolling smoothing-spline denoising, fit on train and re-applied to test.

For each (location, variable) series: try a grid of spline smoothness values
(lambda), pick the one with the lowest rolling-window prediction RMSE on the
training portion, and only denoise if that RMSE clears rmse_threshold
(otherwise the series is left raw). apply_denoise_test replays the chosen
lambda on new data without re-fitting.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
from numpy.lib.stride_tricks import sliding_window_view

WINDOW = 20
LAMBDA_GRID = np.round(np.arange(0.1, 2.01, 0.1), 2)


def make_selected_long(df: pd.DataFrame, selected_dict: dict[str, list[str]]) -> pd.DataFrame:
    """Wide (date, location, col1, col2, ...) -> long (date, location, variable, value),
    keeping only the columns listed per location in selected_dict.
    """
    parts = []
    for loc, cols in selected_dict.items():
        sub = df[df["location"] == loc]
        cols = [c for c in dict.fromkeys(cols) if c in sub.columns]
        if cols:
            parts.append(
                sub[["date", "location"] + cols].melt(
                    id_vars=["date", "location"], value_vars=cols, var_name="variable", value_name="value"
                )
            )

    if not parts:
        return pd.DataFrame(columns=["date", "location", "variable", "value"])
    return pd.concat(parts, ignore_index=True)


def spline_weights(window: int, lam: float, eval_at: int) -> np.ndarray:
    """Linear weights such that `window` values @ weights ~= smoothing-spline value at eval_at."""
    x = np.arange(window)
    eye = np.eye(window)
    weights = np.empty(window)
    for j in range(window):
        weights[j] = make_smoothing_spline(x, eye[j], lam=lam)(eval_at)
    return weights


def precompute_weights(window: int, lambda_grid) -> dict[float, dict[str, np.ndarray]]:
    weight_map = {}
    for lam in map(float, lambda_grid):
        weight_map[lam] = {
            "pred": spline_weights(window, lam, eval_at=window),
            "smooth": spline_weights(window, lam, eval_at=window - 1),
        }
    return weight_map


def calc_rmse(train: np.ndarray, pred_w: np.ndarray, window: int, step: int = 3) -> float:
    train = np.asarray(train, dtype=float)
    if len(train) <= window:
        return np.nan

    windows = sliding_window_view(train, window)[:-1]
    actuals = train[window:]
    if step > 1:
        windows = windows[::step]
        actuals = actuals[::step]

    valid = np.isfinite(windows).all(axis=1) & np.isfinite(actuals)
    if not valid.any():
        return np.nan

    preds = windows[valid] @ pred_w
    actuals = actuals[valid]
    scale = np.max(np.abs(actuals))
    if scale == 0:
        return np.nan

    return np.sqrt(np.mean((preds - actuals) ** 2)) / scale


def select_lambda(train, window: int, lambda_grid, weight_map, step: int = 3):
    scores = []
    for lam in map(float, lambda_grid):
        rmse = calc_rmse(train=train, pred_w=weight_map[lam]["pred"], window=window, step=step)
        if np.isfinite(rmse):
            scores.append((lam, rmse))

    if not scores:
        return None, np.nan
    return min(scores, key=lambda x: x[1])


def rolling_denoise(values, smooth_w: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = values.copy()
    if len(values) < window:
        return out

    windows = sliding_window_view(values, window)
    valid = np.isfinite(windows).all(axis=1)
    idx = np.arange(window - 1, len(values))[valid]
    out[idx] = windows[valid] @ smooth_w
    return out


def _denoise_one(g: pd.DataFrame, window: int, lambda_grid, train_frac: float, rmse_threshold: float, weight_map, step: int):
    g = g.sort_values("date").copy()
    loc = g["location"].iloc[0]
    var = g["variable"].iloc[0]
    values = g["value"].astype(float).to_numpy()

    n_train = int(len(values) * train_frac)
    train = values[:n_train]
    train_std = np.nanstd(train)

    if len(train) <= window or not np.isfinite(train_std) or train_std == 0:
        g["denoised_value"] = values
        return g, [loc, var, None, np.nan, False]

    lam, rmse = select_lambda(train=train, window=window, lambda_grid=lambda_grid, weight_map=weight_map, step=step)
    do_denoise = lam is not None and rmse >= rmse_threshold

    if do_denoise:
        smooth_w = weight_map[float(lam)]["smooth"]
        denoised = rolling_denoise(values, smooth_w, window)
        # spline can overshoot below 0 near runs of zeros; a search-volume index can't be negative
        g["denoised_value"] = np.clip(denoised, 0, None)
    else:
        g["denoised_value"] = values

    return g, [loc, var, lam, rmse, do_denoise]


def do_denoise(
    long_df: pd.DataFrame,
    window: int = WINDOW,
    lambda_grid=LAMBDA_GRID,
    train_frac: float = 0.7,
    rmse_threshold: float = 0.05,
    step: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit denoising on long_df (date, location, variable, value). Returns
    (denoised long df, per-series summary with the chosen lambda/train_rmse/denoised flag).
    """
    weight_map = precompute_weights(window, lambda_grid)

    denoised_parts, summary_rows = [], []
    for _, g in long_df.groupby(["location", "variable"], sort=False):
        denoised_g, summary_row = _denoise_one(g, window, lambda_grid, train_frac, rmse_threshold, weight_map, step)
        denoised_parts.append(denoised_g)
        summary_rows.append(summary_row)

    if not denoised_parts:
        return (
            pd.DataFrame(columns=list(long_df.columns) + ["denoised_value"]),
            pd.DataFrame(columns=["location", "variable", "lambda", "train_rmse", "denoised"]),
        )

    denoised_df = pd.concat(denoised_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows, columns=["location", "variable", "lambda", "train_rmse", "denoised"])
    return denoised_df, summary


def apply_denoise_test(
    test_long_df: pd.DataFrame,
    train_summary: pd.DataFrame,
    window: int = WINDOW,
    value_col: str = "value",
    output_col: str = "denoised_value",
    date_col: str = "date",
    group_cols=("location", "variable"),
    keep_from=None,
) -> pd.DataFrame:
    """Replay the lambda chosen by do_denoise (on train) onto new data, without re-fitting."""
    df = test_long_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(list(group_cols) + [date_col])

    params = train_summary[list(group_cols) + ["lambda", "denoised"]].copy()
    df = df.merge(params, on=list(group_cols), how="left")

    lambdas = df.loc[df["denoised"].eq(True) & df["lambda"].notna(), "lambda"].astype(float).unique()
    smooth_weights = {lam: spline_weights(window, lam, eval_at=window - 1) for lam in lambdas}

    parts = []
    for _, g in df.groupby(list(group_cols), sort=False):
        g = g.sort_values(date_col).copy()
        values = g[value_col].astype(float).to_numpy()
        lam = g["lambda"].iloc[0]
        denoised = g["denoised"].iloc[0]

        if pd.notna(lam) and pd.notna(denoised) and bool(denoised):
            g[output_col] = rolling_denoise(values, smooth_weights[float(lam)], window)
        else:
            g[output_col] = values
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.drop(columns=["lambda", "denoised"], errors="ignore")

    if keep_from is not None:
        out = out[out[date_col] >= pd.to_datetime(keep_from)].copy()

    return out.reset_index(drop=True)
