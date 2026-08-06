"""K-Nearest Neighbors classifier adapter for fraud detection."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.binary_classification.base import ClassifierAdapter, ClassifierConfig, register_classifier


@dataclass
class KNNConfig(ClassifierConfig):
    """Hyperparameters for the K-Nearest Neighbors classifier."""

    algorithm_name = "knn"

    n_neighbors: int = 15
    weights: str = "distance"
    algorithm: str = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: str = "minkowski"
    n_jobs: int = -1

    def to_classifier_config(self) -> dict:
        return {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "metric": self.metric,
            "n_jobs": self.n_jobs,
        }

    @property
    def display_name(self) -> str:
        return "K-Nearest Neighbors"


@register_classifier
class KNNAdapter(ClassifierAdapter[KNNConfig]):
    """Adapts KNeighborsClassifier to the common classifier interface.

    KNN is a useful non-parametric baseline for fraud detection: it makes
    no assumption about the decision boundary's shape, which can help
    catch localized pockets of fraud that don't fit a global linear or
    tree-based split. `weights="distance"` down-weights the influence of
    far neighbors, which helps when fraud points are sparse and scattered
    among legitimate ones. Distance-based methods are sensitive to feature
    scale, so this is wrapped in a pipeline with a StandardScaler. Note
    that KNN has no real "training" cost but can be slow to score at
    inference time on large datasets since it needs a distance search
    against the full training set.
    """

    def __init__(self, config: KNNConfig):
        super().__init__(config)
        self.model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(**config.to_classifier_config()),
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