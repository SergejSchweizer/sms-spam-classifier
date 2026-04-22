import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.utils import (
    add_selected_tfidf_features,
    compute_clustering_metric,
    plot_embeddings_3d_by_label,
    preprocess,
    preprocess_embedding,
)


def test_preprocess_removes_noise_and_stopwords():
    tokens = preprocess("THIS is a FREE msg!!! Visit now")
    assert "is" not in tokens
    assert "a" not in tokens
    assert "free" in tokens
    assert "msg" in tokens


def test_preprocess_embedding_replaces_urls_and_strips_quotes():
    out = preprocess_embedding('check  www.example.com  now  \\"offer\\" ')
    assert "[URL]" in out
    assert '\\"' not in out


def test_add_selected_tfidf_features_returns_prefixed_sparse_columns():
    df = pd.DataFrame(
        {
            "message": [
                "free win now",
                "hello friend",
                "win cash prize",
                "meeting schedule",
            ]
        }
    )
    y = pd.Series([1, 0, 1, 0])

    df_out, selected_cols, _tfidf, _selector = add_selected_tfidf_features(
        df,
        text_col="message",
        target=y,
        min_df=1,
        max_df=1.0,
        max_features=20,
        k=2,
        prefix="tfidf__",
    )

    assert len(selected_cols) == 2
    assert all(col.startswith("tfidf__") for col in selected_cols)
    assert set(selected_cols).issubset(df_out.columns)


def test_compute_clustering_metric_kmeans_inertia_returns_rows():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [3.0, 3.0],
            [3.1, 3.1],
        ]
    )

    result = compute_clustering_metric(
        X,
        method="kmeans",
        metric="inertia",
        param_range=[2, 3],
    )

    assert list(result.columns) == ["param", "value"]
    assert len(result) == 2
    assert result["value"].notna().all()


def test_plot_embeddings_3d_by_label_returns_figure_and_axes():
    df = pd.DataFrame(
        {
            "label": ["ham", "spam", "ham", "spam"],
            "embedding": [
                np.array([0.1, 0.2, 0.3, 0.4]),
                np.array([0.2, 0.1, 0.4, 0.3]),
                np.array([0.9, 0.8, 0.7, 0.6]),
                np.array([0.8, 0.9, 0.6, 0.7]),
            ],
        }
    )

    fig, ax = plot_embeddings_3d_by_label(df)

    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "3D PCA Projection: Ham vs Spam"


def test_plot_embeddings_3d_by_label_missing_columns_raise():
    with pytest.raises(ValueError, match="Missing embedding column"):
        plot_embeddings_3d_by_label(pd.DataFrame({"label": [0, 1]}))
