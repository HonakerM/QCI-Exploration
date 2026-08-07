"""Logistic Regression classifier adapter for fraud detection."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.binary_classification.base import ClassifierAdapter, ClassifierConfig, register_classifier


@dataclass
class LogisticRegressionConfig(ClassifierConfig):
    """Hyperparameters for the Logistic Regression classifier."""

    algorithm_name = "logistic_regression"

    penalty: str | None = "l2"
    C: float = 0.5
    solver: str = "lbfgs"
    max_iter: int = 2000
    class_weight: str|None = "balanced"
    tol: float = 1e-4
    random_state: int = 228

    def to_classifier_config(self) -> dict:
        return {
            "penalty": self.penalty,
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "tol": self.tol,
            "random_state": self.random_state,
        }

    @property
    def display_name(self) -> str:
        return "Logistic Regression"


@register_classifier
class LogisticRegressionAdapter(ClassifierAdapter[LogisticRegressionConfig]):
    """Adapts LogisticRegression to the common classifier interface.

    A fast, highly interpretable linear baseline. It's cheap to train and
    score (useful for latency-sensitive fraud scoring paths), gives
    well-calibrated probabilities that pair naturally with a threshold or
    cost-based decision policy, and `class_weight="balanced"` compensates
    for the low fraud prevalence. Standardizing features matters a lot for
    this model, so it's wrapped in a pipeline with a StandardScaler.
    """

    def __init__(self, config: LogisticRegressionConfig):
        super().__init__(config)
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(**config.to_classifier_config()),
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # Adapter accepts labels in {-1, +1}; sklearn is fine with either,
        # but we map to {0, 1} for consistency with the other adapters.
        y_mapped = np.where(y_train == 1, 1, 0).astype(int)
        self.model.fit(X_train, y_mapped)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X).astype(int)
        return np.where(preds == 1, 1, -1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        joblib.dump(self.model, str(path))