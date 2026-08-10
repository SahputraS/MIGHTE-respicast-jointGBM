"""Deciding which raw keyword columns to keep as-is vs cluster together.

A column with too many zero weeks is unreliable on its own; instead of
dropping it we group it with other sparse, correlated keywords into one
combined query (see cluster_for_location) and re-download that as a single
series in gtrends_download.download_all.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from kneed import KneeLocator


def remove_duplicates(google_df: pd.DataFrame, loc: str, cols: list[str], threshold: float = 0.99) -> list[str]:
    """Drop columns that are near-duplicates (correlation above threshold) of an earlier column."""
    sub = google_df[google_df["location"] == loc][cols]
    corr = sub.corr().abs()
    to_drop = set()
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            if corr.iloc[i, j] > threshold and corr.columns[j] not in to_drop:
                to_drop.add(corr.columns[j])
    return [c for c in cols if c not in to_drop]


def make_zero_df(
    df: pd.DataFrame, zero_cutoff: float = 99, cluster_cutoff: float = 30, dup_threshold: float = 0.99
) -> pd.DataFrame:
    """Per (location, column), compute % zero weeks and classify OK vs CLUSTER.

    Columns with >= zero_cutoff% zeros are discarded outright as unusable.
    Columns with >= cluster_cutoff% zeros are flagged CLUSTER (too sparse to
    use alone); the rest are OK to use as-is.
    """
    value_cols = [c for c in df.columns if c not in ["date", "location"]]

    rows = []
    for loc, df_loc in df.groupby("location"):
        for col in value_cols:
            zero_pct = (df_loc[col] == 0).mean() * 100
            rows.append({"location": loc, "column": col, "zero_pct": zero_pct})

    zero_df = pd.DataFrame(rows)
    zero_df = zero_df[zero_df["zero_pct"] < zero_cutoff].copy()

    cleaned = []
    for loc, group in zero_df.groupby("location"):
        loc_cols = group["column"].tolist()
        if len(loc_cols) > 3:
            loc_cols = remove_duplicates(df, loc, loc_cols, threshold=dup_threshold)
        cleaned.append(group[group["column"].isin(loc_cols)])
    zero_df = pd.concat(cleaned, ignore_index=True)

    zero_df["action"] = zero_df["zero_pct"].apply(lambda x: "CLUSTER" if x >= cluster_cutoff else "OK")
    zero_df = zero_df.sort_values(["location", "zero_pct"], ascending=[True, False]).reset_index(drop=True)

    print(zero_df["action"].value_counts())
    return zero_df


def make_ok_dict(zero_df: pd.DataFrame) -> dict[str, list[str]]:
    """location -> list of columns classified OK (usable without clustering)."""
    return zero_df[zero_df["action"] == "OK"].groupby("location")["column"].apply(list).to_dict()


def cluster_for_location(
    google_df: pd.DataFrame, zero_df: pd.DataFrame, loc: str, plot: bool = True
) -> pd.DataFrame | None:
    """Group a location's sparse (CLUSTER-flagged) keywords via Ward hierarchical
    clustering on correlation distance, with cluster count picked by the Kneedle
    elbow method on within-cluster sum of squares. Returns one row per cluster
    with its '+'-joined term list, ready for gtrends_download.download_all.
    """
    to_cluster = zero_df[(zero_df["location"] == loc) & (zero_df["action"] == "CLUSTER")]["column"].tolist()

    if len(to_cluster) < 2:
        print(f"{loc}: fewer than 2 series to cluster, skipping")
        return None

    sub = google_df[google_df["location"] == loc].sort_values("date")
    ts_matrix = sub[to_cluster].T

    corr = ts_matrix.T.corr().fillna(0).clip(-1, 1)
    dist = (1 - corr).to_numpy(copy=True)
    np.fill_diagonal(dist, 0)
    dist[dist < 0] = 0
    dist_condensed = squareform(dist, checks=False)

    Z = linkage(dist_condensed, method="ward")

    max_k = min(len(to_cluster), 15)
    k_range = list(range(1, max_k + 1))
    wcss = []
    for k in k_range:
        lbls = fcluster(Z, t=k, criterion="maxclust")
        total_wcss = 0
        for c in np.unique(lbls):
            members = ts_matrix.values[lbls == c]
            centroid = members.mean(axis=0)
            total_wcss += ((members - centroid) ** 2).sum()
        wcss.append(total_wcss)

    kneedle = KneeLocator(k_range, wcss, curve="convex", direction="decreasing")
    optimal_k = kneedle.elbow if kneedle.elbow else 1

    labels = fcluster(Z, t=optimal_k, criterion="maxclust")

    unique, counts = np.unique(labels, return_counts=True)
    singletons = unique[counts == 1]
    non_singletons = unique[counts > 1]

    if len(singletons) > 0 and len(non_singletons) > 0:
        centroids = {c: ts_matrix.values[labels == c].mean(axis=0) for c in non_singletons}
        for s in singletons:
            s_idx = np.where(labels == s)[0][0]
            s_series = ts_matrix.values[s_idx]
            best_cluster, best_dist = None, np.inf
            for c, centroid in centroids.items():
                d = 1 - np.corrcoef(s_series, centroid)[0, 1]
                if d < best_dist:
                    best_dist = d
                    best_cluster = c
            labels[s_idx] = best_cluster
            print(f'  {loc}: merged singleton "{to_cluster[s_idx]}" -> cluster {best_cluster}')
    elif len(non_singletons) == 0:
        labels[:] = 1
        print(f"  {loc}: all singletons, merged into 1 cluster")

    cluster_strings = []
    for c in sorted(np.unique(labels)):
        members = [to_cluster[i] for i in range(len(to_cluster)) if labels[i] == c]
        cluster_strings.append(" + ".join(sorted(members)))

    result = pd.DataFrame(
        {
            "location": loc,
            "cluster_id": range(1, len(cluster_strings) + 1),
            "terms": cluster_strings,
            "n_terms": [s.count("+") + 1 for s in cluster_strings],
        }
    )

    if plot:
        n_clusters = len(np.unique(labels))
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        threshold_distance = Z[-(optimal_k - 1), 2] if optimal_k > 1 else Z[-1, 2]
        dendrogram(Z, labels=to_cluster, ax=axes[0], leaf_rotation=90, leaf_font_size=8, color_threshold=threshold_distance)
        axes[0].set_title(f"{loc} — {n_clusters} cluster(s) (after merging singletons)")
        axes[0].axhline(y=threshold_distance, color="r", linestyle="--", label=f"kneedle cut -> {optimal_k}")
        axes[0].legend()

        axes[1].plot(k_range, wcss, "bo-")
        axes[1].axvline(x=optimal_k, color="r", linestyle="--", label=f"elbow = {optimal_k}")
        axes[1].set_xlabel("Number of clusters")
        axes[1].set_ylabel("WCSS")
        axes[1].set_title(f"{loc} — Elbow method")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return result
