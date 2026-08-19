"""Wrap the scikit-learn LDA model in the shared adapter interface."""

from dataclasses import dataclass

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from common.binary_classification.base import ClassifierConfig, register_classifier
from common.binary_classification.classifiers.sklearn import SklearnClassifierAdapter


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
class LDAAdapter(SklearnClassifierAdapter[LDAConfig]):
    """Wrap sklearn LDA in the shared binary-classification adapter API."""

    estimator_cls = LinearDiscriminantAnalysis
