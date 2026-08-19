"""Wrap the scikit-learn logistic regression model in the shared adapter interface."""

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from common.binary_classification.base import ClassifierConfig, register_classifier
from common.binary_classification.classifiers.sklearn import SklearnClassifierAdapter


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
    class_weight: str | None = "balanced"
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
class LogisticRegressionAdapter(SklearnClassifierAdapter[LogisticRegressionConfig]):
    """Wrap sklearn logistic regression in the shared binary-classification adapter API."""

    estimator_cls = LogisticRegression
