import numpy as np
import pandas as pd

from src.training_utils import analyze_errors, evaluate_final_model


class ProbaModel:
    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        n = len(X)
        probs = self._probs[:n]
        return np.column_stack([1.0 - probs, probs])


class PredictOnlyModel:
    def __init__(self, preds):
        self._preds = np.asarray(preds, dtype=int)

    def predict(self, X):
        return self._preds[: len(X)]


def test_evaluate_final_model_with_probabilities_includes_roc_auc():
    model = ProbaModel([0.1, 0.9, 0.8, 0.2])
    X_test = pd.DataFrame({"x": [0, 1, 2, 3]})
    y_test = pd.Series([0, 1, 1, 0])

    metrics = evaluate_final_model(model, X_test, y_test, threshold=0.5)

    assert "roc_auc" in metrics
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)
    assert isinstance(metrics["classification_report"], str)


def test_evaluate_final_model_predict_only_omits_roc_auc():
    model = PredictOnlyModel([0, 1, 0, 1])
    X_test = pd.DataFrame({"x": [0, 1, 2, 3]})
    y_test = pd.Series([0, 1, 1, 0])

    metrics = evaluate_final_model(model, X_test, y_test)

    assert "roc_auc" not in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_analyze_errors_splits_false_negatives_and_positives():
    model = ProbaModel([0.2, 0.4, 0.9, 0.8])
    X_test = pd.DataFrame({"x": [10, 11, 12, 13]})
    y_test = pd.Series([1, 0, 1, 0])
    df_test = pd.DataFrame(
        {"message": ["spam missed", "ham ok", "spam caught", "ham flagged"]}
    )

    false_negatives, false_positives = analyze_errors(
        model,
        X_test,
        y_test,
        df_test,
        text_col="message",
        threshold=0.5,
    )

    assert list(false_negatives["message"]) == ["spam missed"]
    assert list(false_positives["message"]) == ["ham flagged"]
    assert set(false_negatives.columns) == {"message", "y_true", "y_pred", "pred_score"}
    assert set(false_positives.columns) == {"message", "y_true", "y_pred", "pred_score"}
