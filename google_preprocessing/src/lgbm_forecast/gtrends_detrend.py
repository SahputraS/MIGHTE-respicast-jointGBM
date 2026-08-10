"""ADF-based detrending (Djorno et al. 2026 approach).

Per (location, variable) series: run the augmented Dickey-Fuller test with
increasingly flexible deterministic terms ('c' -> 'ct' -> 'ctt'). Stop at
the first regression that's stationary at alpha and remove that
trend; fall back to a first difference if none of them are stationary.

The linear/quadratic trend is fit and removed in log1p space (not raw
scale) so the multiplicative detrend can't blow up dividing by a
near-zero fitted value; apply_detrend_test replays the fitted trend
(in the same log space) onto new data without re-fitting.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def detrend_series(series: pd.Series, alpha: float = 0.05, min_obs: int = 10):
    s = series.astype(float)
    y = s.to_numpy()
    t = np.arange(len(y))

    out = pd.Series(np.nan, index=s.index, dtype=float)

    p_c = p_ct = p_ctt = np.nan
    coef = None
    degree = None

    # adfuller can't handle NaN/inf (e.g. gaps from the outer-merge in load_timeseries_wide);
    # fit only on the finite values, but keep detrending/output aligned to the full series.
    valid = np.isfinite(y)
    if valid.sum() < min_obs:
        return out, "too few observations", p_c, p_ct, p_ctt, coef, degree, np.nan, np.nan

    y_valid, t_valid = y[valid], t[valid]
    last_value = y_valid[-1]  # raw, for the first-difference inverse

    # log-scale mean, used to recentre the detrended series back onto the data scale
    mean_ref = np.log1p(np.clip(y_valid, 0, None)).mean()

    if np.std(y_valid) == 0:
        out[:] = y
        return out, "constant", p_c, p_ct, p_ctt, coef, degree, mean_ref, last_value

    def log_detrend(degree):
        z = np.log1p(np.clip(y, 0, None))  # NaN-preserving over the full length
        c = np.polyfit(t_valid, z[valid], degree)
        trend_log = np.polyval(c, t)
        detrended = np.expm1((z - trend_log) + mean_ref)
        return detrended, c.tolist()

    p_c = adfuller(y_valid, regression="c")[1]
    if p_c < alpha:
        out[:] = y
        return out, "no touch", p_c, p_ct, p_ctt, coef, degree, mean_ref, last_value

    p_ct = adfuller(y_valid, regression="ct")[1]
    if p_ct < alpha:
        degree = 1
        detrended, coef = log_detrend(degree)
        out[:] = detrended
        return out, "linear detrend", p_c, p_ct, p_ctt, coef, degree, mean_ref, last_value

    p_ctt = adfuller(y_valid, regression="ctt")[1]
    if p_ctt < alpha:
        degree = 2
        detrended, coef = log_detrend(degree)
        out[:] = detrended
        return out, "quadratic detrend", p_c, p_ct, p_ctt, coef, degree, mean_ref, last_value

    out[:] = s.diff()
    return out, "first difference", p_c, p_ct, p_ctt, coef, degree, mean_ref, last_value


def do_detrend(
    google_df: pd.DataFrame,
    column: str = "denoised_value",
    output_col: str = "detrended_value",
    group_cols=("location", "variable"),
    date_col: str = "date",
    alpha: float = 0.05,
    min_obs: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit detrending on training data. Returns (df with output_col added, per-series
    summary of the action taken and the fitted params apply_detrend_test needs)."""
    df = google_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(list(group_cols) + [date_col]).copy()
    df[output_col] = np.nan

    summary = []
    for keys, idx in df.groupby(list(group_cols), sort=False).groups.items():
        g = df.loc[idx, column]
        detrended, action, p_c, p_ct, p_ctt, coef, degree, mean_ref, last_value = detrend_series(
            g, alpha=alpha, min_obs=min_obs
        )
        df.loc[idx, output_col] = detrended

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row.update(
            {
                "column": column,
                "action": action,
                "p_c": p_c,
                "p_ct": p_ct,
                "p_ctt": p_ctt,
                "coef": coef,  # log-scale polynomial for linear/quadratic
                "degree": degree,
                "mean_ref": mean_ref,  # log-scale mean
                "train_len": int(len(g)),
                "last_train_value": last_value,
                "n_obs": int(g.notna().sum()),
            }
        )
        summary.append(row)

    return df, pd.DataFrame(summary)


def recheck_stationarity_adf(
    df: pd.DataFrame,
    column: str = "denoised_value",
    group_cols=("location", "variable"),
    date_col: str = "date",
    alpha: float = 0.05,
    min_obs: int = 10,
    autolag: str = "AIC",
) -> pd.DataFrame:
    """Report-only ADF check (same test ladder as do_detrend) — does not modify data."""
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(list(group_cols) + [date_col])

    results = []
    for keys, g in data.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        x_raw = pd.to_numeric(g[column], errors="coerce")
        x = x_raw.dropna().to_numpy(dtype=float)

        row = dict(zip(group_cols, keys))
        row["column_tested"] = column
        row["n_total"] = len(x_raw)
        row["n_used"] = len(x)
        row["n_missing"] = int(x_raw.isna().sum())

        if len(x) < min_obs:
            row.update({"p_c": np.nan, "p_ct": np.nan, "p_ctt": np.nan, "stationarity_status": "too few observations"})
            results.append(row)
            continue

        if np.nanstd(x) == 0:
            row.update({"p_c": np.nan, "p_ct": np.nan, "p_ctt": np.nan, "stationarity_status": "constant series"})
            results.append(row)
            continue

        def safe_adf(y, regression):
            try:
                return adfuller(y, regression=regression, autolag=autolag)[1]
            except Exception:
                return np.nan

        p_c = safe_adf(x, regression="c")
        p_ct = safe_adf(x, regression="ct")
        p_ctt = safe_adf(x, regression="ctt")

        if pd.notna(p_c) and p_c < alpha:
            status = "stationary around constant"
        elif pd.notna(p_ct) and p_ct < alpha:
            status = "trend-stationary: linear"
        elif pd.notna(p_ctt) and p_ctt < alpha:
            status = "trend-stationary: quadratic"
        else:
            status = "non-stationary"

        row.update({"p_c": p_c, "p_ct": p_ct, "p_ctt": p_ctt, "stationarity_status": status})
        results.append(row)

    return pd.DataFrame(results)


def apply_detrend_test(
    test_df: pd.DataFrame,
    detrend_summary: pd.DataFrame,
    column: str = "denoised_value",
    output_col: str = "detrended_value",
    group_cols=("location", "variable"),
    date_col: str = "date",
) -> pd.DataFrame:
    """Replay the trend fitted by do_detrend (on train) onto new data, without re-fitting."""
    df = test_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(list(group_cols) + [date_col])

    params = detrend_summary[list(group_cols) + ["action", "coef", "mean_ref", "train_len", "last_train_value"]].copy()
    df = df.merge(params, on=list(group_cols), how="left")

    def transform_group(g):
        g = g.sort_values(date_col).copy()
        y = pd.to_numeric(g[column], errors="coerce").to_numpy(dtype=float)
        action = g["action"].iloc[0]

        if action == "first difference":
            out = np.full(len(y), np.nan, dtype=float)
            if len(y) > 1:
                out[1:] = np.diff(y)
            ltv = g["last_train_value"].iloc[0]
            if pd.notna(ltv) and len(out) > 0:
                out[0] = y[0] - float(ltv)  # anchor first test diff on last train value

        elif action in ("linear detrend", "quadratic detrend"):
            coef = np.asarray(g["coef"].iloc[0], dtype=float)  # log-scale polynomial
            mean_ref = float(g["mean_ref"].iloc[0])  # log-scale mean
            n_train = int(g["train_len"].iloc[0])
            t = np.arange(n_train, n_train + len(y))  # continue the training index
            z = np.log1p(np.clip(y, 0, None))
            trend_log = np.polyval(coef, t)
            out = np.expm1((z - trend_log) + mean_ref)

        else:  # no touch / constant / missing
            out = y

        g[output_col] = out
        return g

    out = pd.concat([transform_group(g) for _, g in df.groupby(list(group_cols), sort=False)], ignore_index=True)
    return out.drop(columns=["action", "coef", "mean_ref", "train_len", "last_train_value"], errors="ignore")
