import numpy as np
import pandas as pd

from src.modeling import (
    LogisticRegressionStrategy,
    ModelStrategy,
    MultinomialNBStrategy,
    RandomForestStrategy,
    Trainer,
)


class DummyTrial:
    def suggest_float(self, _name, low, _high, log=False):
        return low if log else 0.5

    def suggest_int(self, _name, low, _high):
        return low

    def suggest_categorical(self, _name, choices):
        return choices[0]


def _small_classification_data():
    X = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "f2": [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
            "f3": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return X, y


def test_strategies_build_expected_model_types():
    lr = LogisticRegressionStrategy().build_model()
    rf = RandomForestStrategy().build_model()
    nb = MultinomialNBStrategy().build_model()

    assert lr.__class__.__name__ == "LogisticRegression"
    assert rf.__class__.__name__ == "RandomForestClassifier"
    assert nb.__class__.__name__ == "MultinomialNB"


def test_suggest_params_shapes_are_valid():
    trial = DummyTrial()

    lr_params = LogisticRegressionStrategy().suggest_params(trial)
    rf_params = RandomForestStrategy().suggest_params(trial)
    nb_params = MultinomialNBStrategy().suggest_params(trial)

    assert {"C", "solver", "class_weight", "max_iter"} <= set(lr_params)
    assert {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"} <= set(
        rf_params
    )
    assert {"alpha", "fit_prior"} <= set(nb_params)


def test_trainer_cross_val_recall_returns_bounded_float():
    X, y = _small_classification_data()
    trainer = Trainer(LogisticRegressionStrategy(), n_splits=5, random_state=42)

    score = trainer.cross_val_recall(X, y)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_extract_importance_handles_linear_tree_and_nb_paths():
    strategy = LogisticRegressionStrategy()
    feature_names = ["a", "b", "c"]

    class LinearModel:
        coef_ = np.array([[0.2, -0.9, 0.1]])

    class TreeModel:
        feature_importances_ = np.array([0.1, 0.7, 0.2])

    class NBModel:
        feature_log_prob_ = np.array([[0.0, -1.0, -2.0], [-1.0, -0.3, -0.1]])
        classes_ = np.array([0, 1])

    linear_df = strategy.extract_importance(LinearModel(), feature_names, top_n=2)
    tree_df = strategy.extract_importance(TreeModel(), feature_names, top_n=2)
    nb_df = strategy.extract_importance(NBModel(), feature_names, top_n=2)

    assert list(linear_df.columns) == [
        "feature",
        "coefficient",
        "abs_coefficient",
        "direction",
    ]
    assert list(tree_df.columns) == ["feature", "importance"]
    assert list(nb_df.columns) == [
        "feature",
        "log_prob_diff",
        "abs_log_prob_diff",
        "direction",
    ]
    assert len(linear_df) == 2
    assert len(tree_df) == 2
    assert len(nb_df) == 2


def test_model_strategy_interface_stays_subclassable():
    class MinimalStrategy(ModelStrategy):
        def suggest_params(self, trial):
            return {}

        def build_model(self, params=None):
            return object()

        def name(self):
            return "minimal"

    strategy = MinimalStrategy()
    assert strategy.name() == "minimal"
