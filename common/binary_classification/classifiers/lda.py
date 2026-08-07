"""Linear Discriminant Analysis classifier adapter for fraud detection."""

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
    """Hyperparameters for the Linear Discriminant Analysis classifier."""

    algorithm_name = "lda"

    solver: str = "lsqr"
    shrinkage: str | float | None = "auto"
    tol: float = 1e-4

    def to_classifier_config(self) -> dict:
        return {
            "solver": self.solver,
            "shrinkage": self.shrinkage,
            "tol": self.tol,
        }

    @property
    def display_name(self) -> str:
        return "Linear Discriminant Analysis"


@register_classifier
class LDAAdapter(ClassifierAdapter[LDAConfig]):
    """Adapts LinearDiscriminantAnalysis to the common classifier interface.

    LDA is a lightweight generative linear baseline: it models each class
    as a Gaussian with a shared covariance matrix, which makes it very
    cheap to train and score and gives it a useful, different bias than
    the discriminative models (Logistic Regression) and tree ensembles
    (Random Forest, XGBoost, LightGBM) already in this suite. `solver=
    "lsqr"` with `shrinkage="auto"` (Ledoit-Wolf shrinkage) regularizes
    the covariance estimate, which matters here because fraud is rare and
    the covariance for the minority class would otherwise be estimated
    from very few samples. As with KNN, LDA is scale-sensitive, so it's
    wrapped in a StandardScaler pipeline. Note: shrinkage is only
    supported by the "lsqr" and "eigen" solvers, not "svd".
    """

    def __init__(self, config: LDAConfig):
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