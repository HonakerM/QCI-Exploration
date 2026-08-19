"""Wrap the scikit-learn KNN classifier in the shared adapter interface."""

from dataclasses import dataclass

from sklearn.neighbors import KNeighborsClassifier

from common.binary_classification.base import ClassifierConfig, register_classifier
from common.binary_classification.classifiers.sklearn import SklearnClassifierAdapter


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
class KNNAdapter(SklearnClassifierAdapter[KNNConfig]):
    """Wrap sklearn KNN in the shared binary-classification adapter API."""

    estimator_cls = KNeighborsClassifier
