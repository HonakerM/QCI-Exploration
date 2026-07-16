"""Dataclasses shared by the binary classification training scripts."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
from typing import Any
import numpy as np


@dataclass
class DataConfig:
    """Paths and split settings shared by both scripts.

    Attributes:
        train_file (Optional[Path]): Path to the training CSV file, if any.
        test_file (Optional[Path]): Path to the test CSV file, if any.
        test_size (float): Fraction of the balanced dataset held out for evaluation.
        non_fraud_sample_size (int): Number of non-fraud rows sampled to balance
            the dataset (fraud rows are kept in full).
        random_state (int): Seed used for sampling and the train/test split.
        class_name (str): Name of the target/label column.
        v_feature_names (list[str]): Names of the PCA-transformed input columns.
        engineered_feature_names (list[str]): Names of the aggregate features added on
            top of the V-prefixed columns.
        additional_feature_names (list[str]): Names of any extra raw feature columns
            to include alongside the V and engineered features.
        index_column (str): Name of the row identifier column.
    """

    train_file: Optional[Path] = None
    test_file: Optional[Path] = None

    should_over_sample: bool = True
    model_name_override: str | None = None
    test_size: float = 0.2

    limit_sample_size: bool = False
    non_fraud_sample_size: int = 1000

    random_state: int = 42

    class_name: str = "Class"

    v_feature_names: list[str] = field(default_factory=list)

    engineered_feature_names: list[str] = field(
        default_factory=lambda: [
            "Comp_Sum",
            "Comp_Min",
            "Comp_Max",
            "Comp_Avg",
            "Comp_Std",
            "Comp_Pos",
            "Comp_Neg",
            "Comp_Var",
        ]
    )

    additional_feature_names: list[str] = field(default_factory=list)
    index_column: str = "id"

    @property
    def all_feature_names(self) -> list[str]:
        """Returns the full ordered feature list fed into model input arrays.

        Returns:
            The concatenation of v_feature_names, engineered_feature_names,
            and additional_feature_names, in that order.
        """
        return (
            self.v_feature_names
            + self.engineered_feature_names
            + self.additional_feature_names
        )


@dataclass
class DataSplit:
    """Typed container for the four NumPy arrays produced by prep_data().

    Attributes:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): Test labels.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def __post_init__(self) -> None:
        """Validates that feature and label arrays have matching row counts.

        Returns:
            None.
        """
        assert self.X_train.shape[0] == self.y_train.shape[0], (
            "X_train and y_train row counts must match"
        )
        assert self.X_test.shape[0] == self.y_test.shape[0], (
            "X_test and y_test row counts must match"
        )

    @property
    def n_features(self) -> int:
        """Returns the number of feature columns in the training matrix."""
        return self.X_train.shape[1]

    @property
    def n_train(self) -> int:
        """Returns the number of training rows."""
        return self.X_train.shape[0]

    @property
    def n_test(self) -> int:
        """Returns the number of test rows."""
        return self.X_test.shape[0]


@dataclass
class ClassificationMetrics:
    """Binary classification scores for one data split.

    Attributes:
        split (str): Name of the data split these metrics were computed on, e.g.
            "train" or "test".
        precision (float): Precision score for the positive class.
        recall (float): Recall score for the positive class.
        f1 (float): F1 score for the positive class.
        accuracy (float): Overall accuracy.
        confusion_matrix (np.ndarray): Confusion matrix with shape (2, 2).
    """

    split: str
    precision: float
    recall: float
    f1: float
    accuracy: float
    confusion_matrix: np.ndarray

    def __str__(self) -> str:
        """Returns a formatted multi-line summary of the metrics.

        Returns:
            str: A human-readable string with precision, recall, F1, accuracy,
            and the confusion matrix.
        """
        tag = self.split.capitalize()
        return (
            f"  {tag} precision : {self.precision:.4f}\n"
            f"  {tag} recall    : {self.recall:.4f}\n"
            f"  {tag} F1        : {self.f1:.4f}\n"
            f"  {tag} accuracy  : {self.accuracy:.4f}\n"
            f"  {tag} confusion matrix:\n{self.confusion_matrix}"
        )


@dataclass
class ModelResults:
    """Everything produced by training and evaluating one model.

    Attributes:
        model_name (str): Name of the trained model.
        training_time_seconds (float): Wall-clock training time in seconds.
        fpr (np.ndarray): False positive rates for the ROC curve.
        tpr (np.ndarray): True positive rates for the ROC curve.
        auc (float): Area under the ROC curve.
        log_loss (float): Log loss on the test split.
        train_metrics (ClassificationMetrics): Classification metrics computed on the train split.
        test_metrics (ClassificationMetrics): Classification metrics computed on the test split.
    """

    model_name: str
    training_time_seconds: float

    fpr: np.ndarray
    tpr: np.ndarray
    auc: float

    log_loss: float

    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics

    def to_dict(self) -> dict[str, Any]:
        """Converts these results into a JSON-serializable dictionary.

        Returns:
            dict[str, Any]: A dictionary representation of all fields, with NumPy arrays
            converted to lists.
        """
        return {
            "model_name": self.model_name,
            "training_time_seconds": self.training_time_seconds,
            "fpr": self.fpr.tolist(),
            "tpr": self.tpr.tolist(),
            "auc": self.auc,
            "log_loss": self.log_loss,
            "train_metrics": {
                "split": self.train_metrics.split,
                "precision": self.train_metrics.precision,
                "recall": self.train_metrics.recall,
                "f1": self.train_metrics.f1,
                "accuracy": self.train_metrics.accuracy,
                "confusion_matrix": self.train_metrics.confusion_matrix.tolist(),
            },
            "test_metrics": {
                "split": self.test_metrics.split,
                "precision": self.test_metrics.precision,
                "recall": self.test_metrics.recall,
                "f1": self.test_metrics.f1,
                "accuracy": self.test_metrics.accuracy,
                "confusion_matrix": self.test_metrics.confusion_matrix.tolist(),
            },
        }

    def save(self, path: str | Path, indent: int = 2):
        """Persists the results as JSON so they can be reloaded later.

        Args:
            path (str | Path): Destination file path.
            indent (int): Number of spaces to indent the JSON output.
        """
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelResults":
        """Builds a ModelResults from a dictionary produced by to_dict().

        Args:
            data (dict[str, Any]): Dictionary with the same shape as to_dict()'s output.

        Returns:
            ModelResults: The reconstructed ModelResults instance.
        """
        return cls(
            model_name=data["model_name"],
            training_time_seconds=float(data["training_time_seconds"]),
            fpr=np.asarray(data["fpr"], dtype=float),
            tpr=np.asarray(data["tpr"], dtype=float),
            auc=float(data["auc"]),
            log_loss=float(data["log_loss"]),
            train_metrics=ClassificationMetrics(
                split=data["train_metrics"]["split"],
                precision=float(data["train_metrics"]["precision"]),
                recall=float(data["train_metrics"]["recall"]),
                f1=float(data["train_metrics"]["f1"]),
                accuracy=float(data["train_metrics"]["accuracy"]),
                confusion_matrix=np.asarray(
                    data["train_metrics"]["confusion_matrix"], dtype=int
                ),
            ),
            test_metrics=ClassificationMetrics(
                split=data["test_metrics"]["split"],
                precision=float(data["test_metrics"]["precision"]),
                recall=float(data["test_metrics"]["recall"]),
                f1=float(data["test_metrics"]["f1"]),
                accuracy=float(data["test_metrics"]["accuracy"]),
                confusion_matrix=np.asarray(
                    data["test_metrics"]["confusion_matrix"], dtype=int
                ),
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModelResults":
        """Loads results previously written with save().

        Args:
            path (str | Path): Path to the JSON file to load.

        Returns:
            The loaded ModelResults instance.
        """
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def summary(self) -> str:
        """Returns a formatted multi-line summary of the model results.

        Returns:
            str: A human-readable string with training time, AUC, log loss, and
            both the train and test metrics.
        """
        return (
            f"=== {self.model_name} ===\n"
            f"  Training time : {self.training_time_seconds:.2f}s\n"
            f"  AUC           : {self.auc:.6f}\n"
            f"  Log Loss      : {self.log_loss:.6f}\n"
            f"{self.train_metrics}\n"
            f"{self.test_metrics}"
        )
