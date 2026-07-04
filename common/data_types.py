"""
common/config.py
----------------
Shared configuration dataclasses for the fraud detection scripts.
Model-specific configs live in their respective scripts.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    """Paths and split settings shared by both scripts."""

    train_file: Optional[Path] = None
    test_file: Path = Path("test.csv")

    # Fraction of the balanced dataset held out for evaluation
    test_size: float = 0.2

    # Non-fraud rows sampled to balance the dataset (fraud rows kept in full)
    non_fraud_sample_size: int = 1000

    # Seed used for sampling and train/test split
    random_state: int = 42

    # PCA-transformed columns present in the raw Kaggle dataset
    v_feature_names: list[str] = field(
        default_factory=lambda: [f"V{i}" for i in range(1, 29)]
    )

    # Engineered aggregate features added on top of V1-V28
    engineered_feature_names: list[str] = field(
        default_factory=lambda: [
            "V_Sum", "V_Min", "V_Max", "V_Avg",
            "V_Std", "V_Pos", "V_Neg", "V_Var",
        ]
    )

    @property
    def all_feature_names(self) -> list[str]:
        """Full ordered feature list fed into model input arrays."""
        return self.v_feature_names + self.engineered_feature_names + ["Amount", "Time"]

"""
common/results.py
-----------------
Pure-data dataclasses passed between the data, training, evaluation,
and visualization layers.  No ML logic lives here.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DataSplit:
    """Typed container for the four NumPy arrays produced by prep_data()."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def __post_init__(self) -> None:
        assert self.X_train.shape[0] == self.y_train.shape[0], (
            "X_train and y_train row counts must match"
        )
        assert self.X_test.shape[0] == self.y_test.shape[0], (
            "X_test and y_test row counts must match"
        )

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_train(self) -> int:
        return self.X_train.shape[0]

    @property
    def n_test(self) -> int:
        return self.X_test.shape[0]


@dataclass
class ClassificationMetrics:
    """Binary classification scores for one data split."""

    split: str                      # "train" or "test"
    precision: float
    recall: float
    f1: float
    accuracy: float
    confusion_matrix: np.ndarray    # shape (2, 2)

    def __str__(self) -> str:
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
    """Everything produced by training and evaluating one model."""

    model_name: str
    training_time_seconds: float

    # ROC curve arrays
    fpr: np.ndarray
    tpr: np.ndarray
    auc: float

    log_loss: float

    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics

    def to_dict(self) -> dict[str, Any]:
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

    def save(self, path: str | Path, indent: int = 2) -> None:
        """Persist the results as JSON so they can be reloaded later."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelResults":
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
                confusion_matrix=np.asarray(data["train_metrics"]["confusion_matrix"], dtype=int),
            ),
            test_metrics=ClassificationMetrics(
                split=data["test_metrics"]["split"],
                precision=float(data["test_metrics"]["precision"]),
                recall=float(data["test_metrics"]["recall"]),
                f1=float(data["test_metrics"]["f1"]),
                accuracy=float(data["test_metrics"]["accuracy"]),
                confusion_matrix=np.asarray(data["test_metrics"]["confusion_matrix"], dtype=int),
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModelResults":
        """Load results previously written with `save()`."""
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def summary(self) -> str:
        return (
            f"=== {self.model_name} ===\n"
            f"  Training time : {self.training_time_seconds:.2f}s\n"
            f"  AUC           : {self.auc:.6f}\n"
            f"  Log Loss      : {self.log_loss:.6f}\n"
            f"{self.train_metrics}\n"
            f"{self.test_metrics}"
        )
