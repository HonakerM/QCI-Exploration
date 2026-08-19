"""Wrap the XGBoost classifier in the shared adapter interface."""

from dataclasses import dataclass

from xgboost import XGBClassifier

from common.binary_classification.base import ClassifierConfig, register_classifier
from common.binary_classification.classifiers.sklearn import SklearnClassifierAdapter


@dataclass
class XGBoostConfig(ClassifierConfig):
    """Store the configuration for the XGBoost classifier.

    Attributes:
        algorithm_name (str): Registry key used to resolve this adapter.
        n_estimators (int): Number of boosting rounds.
        min_child_weight (int): Minimum sum of instance weight in a child node.
        max_depth (int): Maximum tree depth.
        learning_rate (float): Step size shrinkage for each boosting step.
        subsample (float): Fraction of samples used per tree.
        colsample_bytree (float): Fraction of features used per tree.
        reg_lambda (float): L2 regularization term.
        reg_alpha (float): L1 regularization term.
        gamma (float): Minimum loss reduction required for a split.
        max_bin (int): Maximum number of bins used for value discretization.
        random_state (int): Random seed.
        objective (str): Objective function used for binary classification.
        tree_method (str): Tree construction algorithm.
        eval_metric (str): Metric used to evaluate boosting iterations.
    """

    algorithm_name = "xgboost"

    n_estimators: int = 3093
    min_child_weight: int = 96
    max_depth: int = 12
    learning_rate: float = 0.07516
    subsample: float = 0.95
    colsample_bytree: float = 0.95
    reg_lambda: float = 1.50
    reg_alpha: float = 1.50
    gamma: float = 1.50
    max_bin: int = 512
    random_state: int = 228
    objective: str = "binary:logistic"
    tree_method: str = "auto"
    eval_metric: str = "auc"

    def to_classifier_config(self) -> dict:
        """Convert the config into XGBClassifier arguments.

        Returns:
            dict: Keyword arguments for the XGBoost model.
        """
        return {
            "n_estimators": self.n_estimators,
            "min_child_weight": self.min_child_weight,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "gamma": self.gamma,
            "max_bin": self.max_bin,
            "random_state": self.random_state,
            "objective": self.objective,
            "tree_method": self.tree_method,
            "eval_metric": self.eval_metric,
        }

    @property
    def display_name(self) -> str:
        """Return the user-facing XGBoost label.

        Returns:
            str: Display name for the model.
        """
        return "XGBoost"


@register_classifier
class XGBoostAdapter(SklearnClassifierAdapter[XGBoostConfig]):
    """Wrap the XGBoost classifier in the shared binary-classification adapter API."""

    estimator_cls = XGBClassifier

    def build_model(self):
        """Create the XGBoost model with categorical support enabled."""
        return self.estimator_cls(
            **self.config.to_classifier_config(),
            enable_categorical=True,
        )

    def save(self, path):
        """Persist the fitted model to disk.

        Args:
            path: Output file for the serialized model.
        """
        self.model.save_model(str(path))
