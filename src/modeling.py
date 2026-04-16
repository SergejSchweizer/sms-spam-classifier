"""Model strategy and training classes for SMS spam analysis.

Strategy pattern overview:
- ``ModelStrategy`` is the common interface (contract) for all models.
- Each concrete strategy encapsulates one algorithm family
  (Logistic Regression, Random Forest, Naive Bayes variants).
- ``Trainer`` depends only on the strategy interface, not on concrete models.

This lets us swap model families without changing training orchestration code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB


class ModelStrategy(ABC):
    """Base strategy interface used by the trainer.

    Why this exists:
    - The trainer should run cross-validation and fitting the same way
      regardless of model type.
    - Each model family can expose its own hyperparameter search space and
      model construction details behind this common API.
    """

    @abstractmethod
    def suggest_params(self, trial) -> Dict[str, Any]:
        """Return Optuna search space suggestions for one trial."""
        pass

    @abstractmethod
    def build_model(self, params: Optional[Dict[str, Any]] = None):
        """Build and return a configured sklearn-compatible model instance."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy identifier used in logs/results."""
        pass

    def extract_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 20,
    ) -> Optional[pd.DataFrame]:
        """Extract feature importance with model-specific fallbacks.

        Supports:
        - linear models via ``coef_``
        - tree models via ``feature_importances_``
        - Naive Bayes via ``feature_log_prob_`` difference (spam vs ham)
        """
        if hasattr(model, "coef_"):
            coefs = np.asarray(model.coef_).ravel()
            df_imp = pd.DataFrame(
                {
                    "feature": feature_names,
                    "coefficient": coefs,
                }
            )
            df_imp["abs_coefficient"] = df_imp["coefficient"].abs()
            df_imp["direction"] = np.where(
                df_imp["coefficient"] >= 0,
                "pushes_to_spam",
                "pushes_to_ham",
            )
            return (
                df_imp.sort_values("abs_coefficient", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )

        if hasattr(model, "feature_importances_"):
            df_imp = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": model.feature_importances_,
                }
            )
            return (
                df_imp.sort_values("importance", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )

        if hasattr(model, "feature_log_prob_"):
            log_probs = np.asarray(model.feature_log_prob_)
            if log_probs.ndim != 2 or log_probs.shape[1] != len(feature_names):
                return None

            classes = getattr(model, "classes_", np.arange(log_probs.shape[0]))
            spam_idx = (
                int(np.where(classes == 1)[0][0])
                if np.any(classes == 1)
                else int(min(1, log_probs.shape[0] - 1))
            )
            ham_idx = int(np.where(classes == 0)[0][0]) if np.any(classes == 0) else 0

            log_prob_diff = log_probs[spam_idx] - log_probs[ham_idx]
            df_imp = pd.DataFrame(
                {
                    "feature": feature_names,
                    "log_prob_diff": log_prob_diff,
                }
            )
            df_imp["abs_log_prob_diff"] = df_imp["log_prob_diff"].abs()
            df_imp["direction"] = np.where(
                df_imp["log_prob_diff"] >= 0,
                "pushes_to_spam",
                "pushes_to_ham",
            )
            return (
                df_imp.sort_values("abs_log_prob_diff", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )

        return None


class LogisticRegressionStrategy(ModelStrategy):
    """Concrete strategy for logistic regression models."""

    def suggest_params(self, trial) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 1e-3, 10, log=True),
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 2000,
        }

    def build_model(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "C": 1.0,
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 2000,
        }
        params = {**defaults, **(params or {})}
        return LogisticRegression(**params)

    def name(self) -> str:
        return "logistic_regression"


class RandomForestStrategy(ModelStrategy):
    """Concrete strategy for random forest models."""

    def suggest_params(self, trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "random_state": 42,
            "n_jobs": -1,
        }

    def build_model(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        params = {**defaults, **(params or {})}
        return RandomForestClassifier(**params)

    def name(self) -> str:
        return "random_forest"


class MultinomialNBStrategy(ModelStrategy):
    """Concrete strategy for Multinomial Naive Bayes."""

    def suggest_params(self, trial) -> Dict[str, Any]:
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "fit_prior": trial.suggest_categorical("fit_prior", [True, False]),
        }

    def build_model(self, params: Optional[Dict[str, Any]] = None):
        defaults = {
            "alpha": 1.0,
            "fit_prior": True,
        }
        params = {**defaults, **(params or {})}
        return MultinomialNB(**params)

    def name(self) -> str:
        return "multinomial_nb"


class Trainer:
    """Training orchestration that works with any ``ModelStrategy``.

    Strategy pattern in action:
    - ``Trainer`` calls ``strategy.suggest_params()`` and ``strategy.build_model()``
      without caring which concrete model is used.
    - Adding a new model means adding a new strategy class only.
    """

    def __init__(self, strategy, n_splits: int = 5, random_state: int = 42):
        self.strategy = strategy
        self.cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

    def cross_val_recall(self, X, y, trial=None):
        """Run stratified CV and return mean recall across folds."""
        recalls = []

        params = self.strategy.suggest_params(trial) if trial is not None else None

        for train_idx, val_idx in self.cv.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            model = self.strategy.build_model(params)
            model.fit(X_train_fold, y_train_fold)

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_val_fold)[:, 1]
                y_pred = (y_proba >= 0.5).astype(int)
            elif hasattr(model, "decision_function"):
                raw_score = model.decision_function(X_val_fold)
                y_proba = 1 / (1 + np.exp(-raw_score))
                y_pred = (y_proba >= 0.5).astype(int)
            else:
                y_pred = model.predict(X_val_fold)

            recalls.append(recall_score(y_val_fold, y_pred))

        return float(np.mean(recalls))


__all__ = [
    "ModelStrategy",
    "LogisticRegressionStrategy",
    "RandomForestStrategy",
    "MultinomialNBStrategy",
    "Trainer",
]
