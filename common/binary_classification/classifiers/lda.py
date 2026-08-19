"""Wrap the scikit-learn LDA model in the shared adapter interface."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.binary_classification.base import ClassifierAdapter, ClassifierConfig, register_classifier


@dataclass
class LDAConfig(ClassifierConfig):
    """Store the configuration for the linear discriminant analysis model.

    Attributes:
        algorithm_name (str): Registry key used to resolve this adapter.
        solver (str): LDA solver to use.
        shrinkage (str | float | None): Shrinkage regularization for the covariance estimate.
        tol (float): Convergence tolerance for the solver.
    """

    algorithm_name = "lda"

    solver: str = "lsqr"
    shrinkage: str | float | None = "auto"
    tol: float = 1e-4

    def to_classifier_config(self) -> dict:
        """Convert the config into sklearn LinearDiscriminantAnalysis arguments.

        Returns:
            dict: Keyword arguments for the sklearn model.
        """
        return {
            "solver": self.solver,
            "shrinkage": self.shrinkage,
            "tol": self.tol,
        }

    @property
    def display_name(self) -> str:
        """Return the user-facing LDA label.

        Returns:
            str: Display name for the model.
        """
        return "Linear Discriminant Analysis"


@register_classifier
class LDAAdapter(ClassifierAdapter[LDAConfig]):
    """Wrap sklearn LDA in the shared binary-classification adapter API."""

    def __init__(self, config: LDAConfig):
        """Create the scaled LDA pipeline.

        Args:
            config (LDAConfig): LDA hyperparameters.
        """
        super().__init__(config)
        self.model = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(**config.to_classifier_config()),
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