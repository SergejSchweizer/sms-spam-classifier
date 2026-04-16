"""Training, evaluation, and reporting helpers for SMS spam analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from .modeling import ModelStrategy, Trainer


def optimize_strategy(
    strategy,
    X,
    y,
    n_trials: int = 30,
    study_name: Optional[str] = None,
):
    """Run Optuna optimization for a strategy using CV recall as objective."""
    # Trainer centralizes CV behavior so optimization logic stays strategy-agnostic.
    trainer = Trainer(strategy)
    pbar = tqdm(total=n_trials, desc=f"Optimizing {strategy.name()}", unit="trial")

    def objective(trial):
        # Each trial asks the strategy for params, then scores mean CV recall.
        return trainer.cross_val_recall(X, y, trial)

    def tqdm_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        # Surface both current-trial and best-so-far recall to make search progress visible.
        best_value = study.best_value if study.best_trial is not None else None

        if best_value is not None:
            pbar.set_postfix(
                {
                    "best_recall": f"{best_value:.4f}",
                    "last_recall": f"{trial.value:.4f}",
                }
            )
        else:
            pbar.set_postfix({"last_recall": f"{trial.value:.4f}"})

        pbar.update(1)

    study = optuna.create_study(direction="maximize", study_name=study_name)

    try:
        # Run optimization with a callback so progress bar stays in sync with trial completions.
        study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback])
    finally:
        # Always close the progress bar to keep notebook/terminal output clean.
        pbar.close()

    return study


def train_final_model(strategy, best_params, X, y):
    """Fit one final model instance on all training data."""
    # Rebuild from best hyperparameters before the final full-data fit.
    model = strategy.build_model(best_params)
    model.fit(X, y)
    return model


def evaluate_final_model(
    model: Any,
    X_test: Any,
    y_test: pd.Series | np.ndarray,
    threshold: float = 0.5,
    positive_label: int = 1,
) -> Dict[str, Any]:
    """Evaluate trained model and return a metrics dictionary."""
    # Normalize y_test shape/type once so metric functions behave consistently.
    y_true = np.asarray(y_test)

    if hasattr(model, "predict_proba"):
        # Preferred path for probabilistic classifiers.
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= threshold).astype(int)
    elif hasattr(model, "decision_function"):
        # Convert raw decision scores to pseudo-probabilities for thresholding and AUC.
        raw_score = model.decision_function(X_test)
        y_score = 1 / (1 + np.exp(-raw_score))
        y_pred = (y_score >= threshold).astype(int)
    else:
        # Fallback for estimators that expose only hard predictions.
        y_pred = model.predict(X_test)
        y_score = None

    # Keep both scalar metrics and richer diagnostic artifacts.
    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

    if y_score is not None:
        # ROC-AUC needs continuous scores, so expose it only when available.
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)

    return metrics


def analyze_errors(
    model: Any,
    X_test: Any,
    y_test: pd.Series | np.ndarray,
    df_test: pd.DataFrame,
    text_col: str = "message",
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return false negatives and false positives sorted by model confidence."""
    # Keep a dense 1D ground-truth array for vectorized comparisons.
    y_true = np.asarray(y_test)

    if hasattr(model, "predict_proba"):
        # Use predicted probability to rank mistakes by model confidence.
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= threshold).astype(int)
    else:
        # When no probabilities exist, still return errors with NaN score placeholders.
        y_pred = model.predict(X_test)
        y_score = np.full(shape=len(y_pred), fill_value=np.nan)

    # Build one aligned results table for convenient filtering.
    results = df_test.copy().reset_index(drop=True)
    results["y_true"] = y_true
    results["y_pred"] = y_pred
    results["pred_score"] = y_score

    # Keep only requested text + prediction columns when available.
    keep_cols = [
        col for col in [text_col, "y_true", "y_pred", "pred_score"] if col in results.columns
    ]

    # False negatives: true spam predicted as ham, sorted by lowest spam score.
    false_negatives = (
        results[(results["y_true"] == 1) & (results["y_pred"] == 0)][keep_cols]
        .sort_values("pred_score", ascending=True, na_position="last")
        .reset_index(drop=True)
    )

    # False positives: true ham predicted as spam, sorted by highest spam score.
    false_positives = (
        results[(results["y_true"] == 0) & (results["y_pred"] == 1)][keep_cols]
        .sort_values("pred_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    return false_negatives, false_positives


def run_strategy_pipeline(
    strategy: ModelStrategy,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    text_col: str,
    feature_names: List[str],
    n_trials: int = 30,
    top_n: int = 20,
) -> Dict[str, Any]:
    """Run full strategy workflow: tuning, fitting, evaluation, and analysis."""
    # 1) Hyperparameter search driven by CV recall.
    study = optimize_strategy(strategy, X_train, y_train, n_trials=n_trials)

    # 2) Train a final estimator on all training data with best discovered params.
    final_model = train_final_model(strategy, study.best_params, X_train, y_train)

    # 3) Evaluate generalization quality on the held-out test split.
    final_model_metrics = evaluate_final_model(final_model, X_test, y_test)

    # 4) Extract high-value error slices for qualitative review.
    false_negatives, false_positives = analyze_errors(
        final_model,
        X_test,
        y_test,
        df_test,
        text_col=text_col,
    )

    # 5) Pull top feature contributions when the model family supports it.
    final_model_importance = strategy.extract_importance(
        final_model,
        feature_names,
        top_n=top_n,
    )

    return {
        "strategy_name": strategy.name(),
        "study": study,
        "best_params": study.best_params,
        "model": final_model,
        "metrics": final_model_metrics,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "importance": final_model_importance,
    }


__all__ = [
    "optimize_strategy",
    "train_final_model",
    "run_strategy_pipeline",
    "evaluate_final_model",
    "analyze_errors",
]
