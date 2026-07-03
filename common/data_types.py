"""
common/config.py
----------------
Shared configuration dataclasses for the fraud detection scripts.
Model-specific configs live in their respective scripts.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Paths and split settings shared by both scripts."""

    train_file: Path = Path("train.csv")
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

from dataclasses import dataclass

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

    def summary(self) -> str:
        return (
            f"=== {self.model_name} ===\n"
            f"  Training time : {self.training_time_seconds:.2f}s\n"
            f"  AUC           : {self.auc:.6f}\n"
            f"  Log Loss      : {self.log_loss:.6f}\n"
            f"{self.train_metrics}\n"
            f"{self.test_metrics}"
        )
