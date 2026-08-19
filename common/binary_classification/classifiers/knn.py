"""Wrap the scikit-learn KNN classifier in the shared adapter interface."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.binary_classification.base import (
    ClassifierAdapter,
    ClassifierConfig,
    register_classifier,
)


@dataclass
class KNNConfig(ClassifierConfig):
    """Store the configuration for the K-nearest neighbors classifier.

    Attributes:
        algorithm_name (str): Registry key used to resolve this adapter.
        n_neighbors (int): Number of neighbors considered for each prediction.
        weights (str): Neighbor weighting strategy.
        algorithm (str): Search algorithm passed to sklearn.
        leaf_size (int): Leaf size for tree-based search.
        p (int): Power parameter for the Minkowski metric.
        metric (str): Distance metric used by KNN.
        n_jobs (int): Parallelism used by the scikit-learn model.
    """

    algorithm_name = "knn"

    n_neighbors: int = 15
    weights: str = "distance"
    algorithm: str = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: str = "minkowski"
    n_jobs: int = -1

    def to_classifier_config(self) -> dict:
        """Convert the config into sklearn KNeighborsClassifier arguments.

        Returns:
            dict: Keyword arguments for the sklearn model.
        """
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
        """Return the model label used in reports and plots.

        Returns:
            str: User-facing display name.
        """
        return "K-Nearest Neighbors"


@register_classifier
class KNNAdapter(ClassifierAdapter[KNNConfig]):
    """Wrap sklearn KNN in the shared binary-classification adapter API."""

    def __init__(self, config: KNNConfig):
        """Create the fitted pipeline for the KNN model.

        Args:
            config (KNNConfig): KNN hyperparameters.
        """
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
