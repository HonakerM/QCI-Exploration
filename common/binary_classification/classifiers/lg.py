"""Wrap the scikit-learn logistic regression model in the shared adapter interface."""

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
    """Store the configuration for the logistic regression model.

    Attributes:
        algorithm_name (str): Registry key used to resolve this adapter.
        penalty (str | None): Regularization penalty applied to the coefficients.
        C (float): Inverse regularization strength.
        solver (str): Optimization solver used by sklearn.
        max_iter (int): Maximum number of optimization iterations.
        class_weight (str | None): Weighting strategy for the minority class.
        tol (float): Tolerance for convergence.
        random_state (int): Seed used by the solver.
    """

    algorithm_name = "logistic_regression"

    penalty: str | None = "l2"
    C: float = 0.5
    solver: str = "lbfgs"
    max_iter: int = 2000
    class_weight: str|None = "balanced"
    tol: float = 1e-4
    random_state: int = 228

    def to_classifier_config(self) -> dict:
        """Convert the config into sklearn LogisticRegression arguments.

        Returns:
            dict: Keyword arguments for the sklearn model.
        """
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
        """Return the user-facing logistic regression label.

        Returns:
            str: Display name for the model.
        """
        return "Logistic Regression"


@register_classifier
class LogisticRegressionAdapter(ClassifierAdapter[LogisticRegressionConfig]):
    """Wrap sklearn logistic regression in the shared binary-classification adapter API."""

    def __init__(self, config: LogisticRegressionConfig):
        """Create the scaled logistic regression pipeline.

        Args:
            config (LogisticRegressionConfig): Model hyperparameters.
        """
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