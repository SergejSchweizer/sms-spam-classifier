"""Shared text and feature-engineering utilities for SMS spam analysis."""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import silhouette_score
from tqdm.notebook import tqdm

STOPWORDS = frozenset(
    {
        "a",
        "the",
        "is",
        "are",
        "i",
        "you",
        "me",
        "my",
        "we",
        "to",
        "and",
        "of",
        "in",
        "on",
        "for",
        "it",
        "this",
        "that",
        "be",
        "have",
        "am",
        "was",
        "so",
        "do",
        "did",
        "but",
        "if",
        "or",
        "as",
        "at",
        "with",
        "u",
    }
)
# Small baseline stopword list for lightweight token filtering.

_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]")
_LT_GT_PATTERN = re.compile(r"(ltgt|lt|gt)")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_URL_PATTERN = re.compile(r"http\S+|www\S+")


def preprocess(text: str) -> list[str]:
    """Lowercase text, remove punctuation, tokenize, and drop stopwords."""
    # Normalize casing and strip non-alphanumeric symbols so token matching is stable.
    text = text.lower()
    text = _NON_ALNUM_PATTERN.sub("", text)
    text = _LT_GT_PATTERN.sub("", text)

    # Keep tokenization lightweight; downstream steps do not require linguistic parsing.
    tokens = text.split()

    # Return only content-bearing terms to reduce noise in n-gram and TF-IDF features.
    return [token for token in tokens if token not in STOPWORDS]


def preprocess_embedding(text: str) -> str:
    """Lightweight text cleanup for transformer-based embedding generation."""
    # Ensure non-string inputs (e.g., NaN/None) are safely handled.
    text = str(text)

    # Collapse repeated spacing and replace URLs with a stable placeholder token.
    text = _WHITESPACE_PATTERN.sub(" ", text)
    text = _URL_PATTERN.sub("[URL]", text)

    # Remove quote noise that often comes from CSV escaping.
    text = text.replace('\\"', "").replace("'", "")
    return text.strip()


def top_ngram_summary(df: pd.DataFrame, n: int = 2, top_k: int = 25) -> pd.DataFrame:
    """Build per-class top-n n-gram summary table with relative frequencies."""
    # Build class-wise n-gram counts in one chained pipeline to keep transformations explicit.
    result = (
        df.assign(
            tokens=df["message"].apply(preprocess),
            ngram=lambda x: x["tokens"].apply(
                lambda toks: [" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)]
            )
        )
        .explode("ngram")
        .dropna(subset=["ngram"])
        .groupby(["label", "ngram"])
        .size()
        .rename("count")
        .reset_index()
        .assign(
            rel_count=lambda x: x["count"] / x.groupby("label")["count"].transform("sum")
        )
        .sort_values(["label", "rel_count"], ascending=[True, False])
        .groupby("label")
        .head(top_k)
        .assign(rank=lambda x: x.groupby("label").cumcount())
        .pivot(index="rank", columns="label", values=["ngram", "count", "rel_count"])
    )

    # Flatten MultiIndex columns from pivot for easier downstream usage.
    result.columns = [f"{label}_{metric}" for metric, label in result.columns]

    # Reorder to a predictable schema and sort by strongest relative signals first.
    result = result[
        [
            "spam_ngram", "spam_count", "spam_rel_count",
            "ham_ngram", "ham_count", "ham_rel_count"
        ]
    ].sort_values(
        by=["spam_rel_count", "ham_rel_count"],
        ascending=False
    ).reset_index(drop=True)

    return result


def add_selected_tfidf_features(
    df: pd.DataFrame,
    text_col: str,
    target: pd.Series,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.90,
    max_features=3000,
    k=300,
    prefix="tfidf__"
) -> tuple[pd.DataFrame, list[str], TfidfVectorizer, SelectKBest]:
    """Create TF-IDF features and keep only top-k by chi-square relevance."""
    # Fit TF-IDF on the chosen text column using bounded vocabulary settings.
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features
    )

    # Represent text in sparse n-gram space.
    X_tfidf = tfidf.fit_transform(df[text_col].fillna(""))

    # Keep only the most target-informative sparse features.
    selector = SelectKBest(score_func=chi2, k=k)
    X_sel = selector.fit_transform(X_tfidf, target)

    # Name retained sparse features and prefix them to avoid column-name collisions.
    selected_names = tfidf.get_feature_names_out()[selector.get_support()]
    selected_cols = [f"{prefix}{name}" for name in selected_names]

    # Keep the sparse representation sparse to avoid memory blow-ups on wider vocabularies.
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(
        X_sel,
        index=df.index,
        columns=selected_cols
    )

    # Return an augmented frame plus fitted transformers for train/test reuse.
    df_out = pd.concat([df, tfidf_df], axis=1)

    return df_out, selected_cols, tfidf, selector


def compute_clustering_metric(
    X,
    method: str = "kmeans",
    metric: str = "silhouette",
    param_range=None,
    dbscan_params=None,
) -> pd.DataFrame:
    """Evaluate clustering quality across KMeans/DBSCAN parameter grids."""
    # Collect each trial's parameter setting and metric value in a tidy table.
    rows = []

    if method == "kmeans":
        for k in tqdm(param_range, desc=f"KMeans {metric}"):
            # Fix random seed for reproducible cluster assignments across runs.
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(X)

            if metric == "inertia":
                value = model.inertia_
            elif metric == "silhouette":
                # Silhouette is undefined for a single cluster; return None in that case.
                value = silhouette_score(X, labels) if len(set(labels)) > 1 else None
            else:
                raise ValueError("Unknown metric")

            rows.append({"param": k, "value": value})

    elif method == "dbscan":
        if dbscan_params is None:
            raise ValueError("dbscan_params required")

        for eps, min_samples in tqdm(dbscan_params, desc="DBSCAN silhouette"):
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)

            # DBSCAN may produce only noise or one cluster; silhouette is invalid then.
            unique = set(labels)
            if len(unique) <= 1 or (len(unique) == 2 and -1 in unique):
                value = None
            else:
                # Exclude noise points when computing silhouette for DBSCAN clusters.
                mask = labels != -1
                if len(set(labels[mask])) < 2:
                    value = None
                else:
                    value = silhouette_score(X[mask], labels[mask])

            rows.append({"param": (eps, min_samples), "value": value})

    else:
        raise ValueError("Unknown method")

    return pd.DataFrame(rows)


def plot_clustering_metric(df: pd.DataFrame, title: str = "Metric") -> None:
    """Plot a clustering metric dataframe generated by compute_clustering_metric."""
    # Build a simple line plot that works for both scalar and tuple-valued parameters.
    plt.figure()

    x = df["param"]
    y = df["value"]

    if isinstance(x.iloc[0], tuple):
        # For DBSCAN grids, use string labels so (eps, min_samples) stays readable.
        x_labels = [str(i) for i in x]
        plt.xticks(rotation=45)
        plt.plot(x_labels, y)
        plt.xlabel("params (eps, min_samples)")
    else:
        # For KMeans, the x-axis is directly the number of clusters k.
        plt.plot(x, y)
        plt.xlabel("k")

    plt.ylabel("score")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_kmeans_clusters_3d(
    X,
    df: pd.DataFrame,
    n_clusters: int = 7,
    text_col: str = "message",
    label_col: str = "label",
    random_state: int = 42,
):
    """Project embeddings to 3D with PCA and visualize KMeans clusters in Matplotlib."""
    # Convert the embedding matrix to a dense numeric array for PCA and KMeans.
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array-like matrix")

    if X.shape[1] < 3:
        raise ValueError("X must have at least 3 features for 3D projection")

    # Create a 3D representation that is easier to visualize than the original embedding space.
    X_3d = PCA(n_components=3, random_state=random_state).fit_transform(X)

    # Fit KMeans in the reduced space so every point receives one cluster assignment.
    cluster_labels = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    ).fit_predict(X_3d)

    # Attach cluster ids and optional hover context from the source dataframe.
    plot_df = pd.DataFrame(
        {
            "pc1": X_3d[:, 0],
            "pc2": X_3d[:, 1],
            "pc3": X_3d[:, 2],
            "cluster": cluster_labels.astype(str),
        }
    )

    if text_col in df.columns:
        plot_df[text_col] = df[text_col].values

    if label_col in df.columns:
        plot_df[label_col] = df[label_col].values

    # Use a standard 3D scatter plot so the visualization works without Plotly.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        plot_df["pc1"],
        plot_df["pc2"],
        plot_df["pc3"],
        c=cluster_labels,
        cmap="tab10",
        s=18,
        alpha=0.75,
    )
    ax.set_title(f"3D PCA Projection with {n_clusters} KMeans Clusters")
    ax.set_xlabel("pc1")
    ax.set_ylabel("pc2")
    ax.set_zlabel("pc3")
    ax.legend(*scatter.legend_elements(), title="cluster", loc="upper right")
    plt.tight_layout()
    return fig, ax


__all__ = [
    "STOPWORDS",
    "preprocess",
    "preprocess_embedding",
    "top_ngram_summary",
    "add_selected_tfidf_features",
    "compute_clustering_metric",
    "plot_clustering_metric",
    "plot_kmeans_clusters_3d",
]
