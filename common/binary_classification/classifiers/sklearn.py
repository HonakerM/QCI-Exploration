"""Shared sklearn adapter logic for binary classifiers."""

from pathlib import Path
from typing import Generic, TypeVar

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.binary_classification.base import ClassifierAdapter, ClassifierConfig

TConfig = TypeVar("TConfig", bound=ClassifierConfig)


class SklearnClassifierAdapter(ClassifierAdapter[TConfig], Generic[TConfig]):
    """Base class for scikit-learn classifiers that use a StandardScaler pipeline."""

    estimator_cls: type

    def __init__(self, config: TConfig):
        super().__init__(config)
        self.model = self.build_model()

    def build_model(self):
        """Create the sklearn pipeline used by this adapter.

        Returns:
            object: Fitted sklearn pipeline.
        """
        return make_pipeline(
            StandardScaler(),
            self.estimator_cls(**self.config.to_classifier_config()),
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the wrapped sklearn model on the provided labels.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels in {-1, 1} format.
        """
        y_mapped = np.where(y_train == 1, 1, 0).astype(int)
        self.model.fit(X_train, y_mapped)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions in the shared {-1, 1} format.

        Args:
            X (np.ndarray): Feature rows to score.

        Returns:
            np.ndarray: Predicted labels.
        """
        preds = self.model.predict(X).astype(int)
        return np.where(preds == 1, 1, -1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the positive-class probability for each row.

        Args:
            X (np.ndarray): Feature rows to score.

        Returns:
            np.ndarray: Positive-class probabilities.
        """
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        """Persist the sklearn pipeline to disk.

        Args:
            path (Path): Output file for the serialized model.
        """
        joblib.dump(self.model, str(path))


__all__ = ["SklearnClassifierAdapter"]
