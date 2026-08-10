"""Visualizations for the clustering and denoise/detrend preprocessing stages."""

import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_location_cluster(
    google_df: pd.DataFrame,
    google_cluster_df: pd.DataFrame,
    cluster_all: pd.DataFrame,
    loc: str = "AT",
    cluster_id: int = 1,
    date_col: str = "date",
    ncols: int = 3,
    figsize_per_panel=(4, 2.6),
    cluster_height: float = 3.5,
) -> list[str]:
    """Individual keyword series stacked above the combined cluster series they were merged into."""
    cluster_name = f"cluster_{cluster_id}"

    row = cluster_all[(cluster_all["location"].eq(loc)) & (cluster_all["cluster_id"].eq(cluster_id))]
    if row.empty:
        raise ValueError(f"No cluster_id={cluster_id} found for location={loc}")

    terms = row["terms"].iloc[0].split(" + ")
    terms = [t for t in terms if t in google_df.columns]
    if len(terms) == 0:
        raise ValueError(f"No individual cluster terms found in google_df for {loc}")

    indiv = google_df[google_df["location"].eq(loc)].copy()
    clust = google_cluster_df[google_cluster_df["location"].eq(loc)].copy()
    indiv[date_col] = pd.to_datetime(indiv[date_col])
    clust[date_col] = pd.to_datetime(clust[date_col])
    indiv = indiv.sort_values(date_col)
    clust = clust.sort_values(date_col)

    if cluster_name not in clust.columns:
        raise ValueError(f"{cluster_name} not found in google_cluster_df")

    n_terms = len(terms)
    n_indiv_rows = math.ceil(n_terms / ncols)
    total_rows = n_indiv_rows + 1

    fig_w = ncols * figsize_per_panel[0]
    fig_h = n_indiv_rows * figsize_per_panel[1] + cluster_height

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(total_rows, ncols, height_ratios=[1] * n_indiv_rows + [1.6], hspace=0.55, wspace=0.25)

    for i, term in enumerate(terms):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.vlines(indiv[date_col], 0, indiv[term], color="orange", linewidth=1)
        ax.set_title(term)
        ax.set_xlabel("Date")
        ax.set_ylabel("Index")
        ax.grid(alpha=0.3)

    for j in range(n_terms, n_indiv_rows * ncols):
        r, c = divmod(j, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    ax_cluster = fig.add_subplot(gs[n_indiv_rows, :])
    ax_cluster.plot(clust[date_col], clust[cluster_name], color="steelblue", linewidth=1)
    ax_cluster.set_title(f"{loc} — {cluster_name}")
    ax_cluster.set_xlabel("Date")
    ax_cluster.set_ylabel("Index")
    ax_cluster.grid(alpha=0.3)

    fig.suptitle(f"{loc}: individual keywords in {cluster_name} vs clustered query", y=0.995)
    plt.show()

    return terms


def plot_preprocessing_single(
    df: pd.DataFrame,
    location: str,
    variable: str,
    date_col: str = "date",
    location_col: str = "location",
    variable_col: str = "variable",
    original_col: str = "value",
    denoised_col: str = "denoised_value",
    detrended_col: str = "detrended_value",
    figsize=(12, 8),
) -> None:
    """Original / denoised / detrended stacked for one (location, variable)."""
    g = df.loc[(df[location_col].eq(location)) & (df[variable_col].eq(variable))].copy()
    if g.empty:
        raise ValueError(f"No data found for location={location}, variable={variable}")

    g[date_col] = pd.to_datetime(g[date_col])
    g = g.sort_values(date_col)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    plot_info = [(original_col, "Original"), (denoised_col, "Denoised"), (detrended_col, "Detrended")]

    for ax, (col, title) in zip(axes, plot_info):
        if col not in g.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

        ax.plot(g[date_col], g[col], linewidth=1)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"{location} — {variable}", y=0.98)
    fig.tight_layout()
    plt.show()


def plot_gt_preprocessing_panel(
    google_df: pd.DataFrame,
    google_cluster_df: pd.DataFrame,
    cluster_all: pd.DataFrame,
    preprocessed_df: pd.DataFrame,
    cluster_loc: str = "AT",
    cluster_id: int = 1,
    preproc_loc: str = "BE",
    preproc_var: str = "nausea",
    date_col: str = "date",
    ncols_cluster: int = 3,
    save_path: str | None = None,
    dpi: int = 200,
) -> None:
    """Two-panel figure: (a) clustering for one location/cluster, (b) original/denoised/detrended
    for one (location, variable) — used for report/paper figures combining both stages."""
    cluster_name = f"cluster_{cluster_id}"

    row = cluster_all[(cluster_all["location"].eq(cluster_loc)) & (cluster_all["cluster_id"].eq(cluster_id))]
    terms = row["terms"].iloc[0].split(" + ")
    terms = [t for t in terms if t in google_df.columns]

    indiv = google_df[google_df["location"].eq(cluster_loc)].copy()
    clust = google_cluster_df[google_cluster_df["location"].eq(cluster_loc)].copy()
    indiv[date_col] = pd.to_datetime(indiv[date_col])
    clust[date_col] = pd.to_datetime(clust[date_col])
    indiv = indiv.sort_values(date_col)
    clust = clust.sort_values(date_col)

    n_terms = len(terms)
    n_indiv_rows = math.ceil(n_terms / ncols_cluster)

    g = preprocessed_df.loc[
        (preprocessed_df["location"].eq(preproc_loc)) & (preprocessed_df["variable"].eq(preproc_var))
    ].copy()
    g[date_col] = pd.to_datetime(g[date_col])
    g = g.sort_values(date_col)

    left_rows = n_indiv_rows + 1
    right_rows = 3
    total_rows = max(left_rows, right_rows)

    fig = plt.figure(figsize=(20, 3.2 * total_rows))
    gs = GridSpec(total_rows, 2, figure=fig, wspace=0.25, hspace=0.5, width_ratios=[1, 1])

    for i, term in enumerate(terms):
        r, c_sub = divmod(i, ncols_cluster)
        gs_left = gs[r, 0].subgridspec(1, ncols_cluster, wspace=0.35)
        ax = fig.add_subplot(gs_left[0, c_sub])
        ax.vlines(indiv[date_col], 0, indiv[term], color="#E8A33D", linewidth=0.8)
        ax.set_title(term, fontsize=10)
        ax.set_ylabel("Index", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    for j in range(n_terms, n_indiv_rows * ncols_cluster):
        r, c_sub = divmod(j, ncols_cluster)
        gs_left = gs[r, 0].subgridspec(1, ncols_cluster, wspace=0.35)
        ax = fig.add_subplot(gs_left[0, c_sub])
        ax.axis("off")

    ax_cl = fig.add_subplot(gs[n_indiv_rows, 0])
    ax_cl.plot(clust[date_col], clust[cluster_name], color="steelblue", linewidth=1)
    ax_cl.set_title(f"{cluster_loc} — {cluster_name}", fontsize=11)
    ax_cl.set_ylabel("Index", fontsize=9)
    ax_cl.set_xlabel("Date", fontsize=9)
    ax_cl.grid(alpha=0.2)

    for extra in range(left_rows, total_rows):
        fig.add_subplot(gs[extra, 0]).axis("off")

    stages = [("value", "Original"), ("denoised_value", "Denoised"), ("detrended_value", "Detrended")]
    for i, (col, title) in enumerate(stages):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(g[date_col], g[col], linewidth=1, color="steelblue")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Index", fontsize=9)
        ax.grid(alpha=0.2)
        if i == len(stages) - 1:
            ax.set_xlabel("Date", fontsize=9)

    for extra in range(right_rows, total_rows):
        fig.add_subplot(gs[extra, 1]).axis("off")

    fig.text(0.02, 0.98, "(a)", fontsize=16, fontweight="bold", va="top")
    fig.text(0.52, 0.98, "(b)", fontsize=16, fontweight="bold", va="top")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    plt.show()
