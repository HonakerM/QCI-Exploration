from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Paths and split settings for data loading."""
    test_path: Path
    train_path: Path
    # Fraction of the balanced dataset held out for evaluation
    test_size: float = 0.2

    # Number of non-fraud rows sampled to balance the dataset
    non_fraud_sample_size: int = 1000

    # Random seed used for train/test split
    random_state: int = 42

    # PCA feature columns present in the raw Kaggle dataset
    v_feature_names: list[str] = field(default_factory=lambda: [
        f"V{i}" for i in range(1, 29)
    ])

    # Engineered aggregate features added on top of V1-V28
    engineered_feature_names: list[str] = field(default_factory=lambda: [
        "V_Sum", "V_Min", "V_Max", "V_Avg", "V_Std", "V_Pos", "V_Neg", "V_Var",
    ])

    @property
    def all_feature_names(self) -> list[str]:
        """Full ordered feature list used for model input arrays."""
        return (
            self.v_feature_names
            + self.engineered_feature_names
            + ["Amount", "Time"]
        )




from dataclasses import dataclass
import numpy as np



@dataclass
class ClassificationMetrics:
    """Binary classification scores for one split (train or test)."""
    split: str          # "train" or "test"
    precision: float
    recall: float
    f1: float
    accuracy: float
    confusion_matrix: np.ndarray   # shape (2, 2), labels [-1, 1]

    def __str__(self) -> str:
        lines = [
            f"  {self.split.capitalize()} precision : {self.precision:.4f}",
            f"  {self.split.capitalize()} recall    : {self.recall:.4f}",
            f"  {self.split.capitalize()} F1        : {self.f1:.4f}",
            f"  {self.split.capitalize()} accuracy  : {self.accuracy:.4f}",
            f"  {self.split.capitalize()} confusion matrix:\n{self.confusion_matrix}",
        ]
        return "\n".join(lines)

@dataclass
class ModelResults:
    """
    Everything produced by training and evaluating one model.

    The fpr / tpr arrays are the ROC curve coordinates.
    Labels in y_* arrays follow the convention set by prep_data:
      - CVQBoost / QBoost: {-1, 1}
      - XGBoost:           {0, 1}  (converted before training)
    """
    model_name: str
    training_time_seconds: float

    # ROC curve
    fpr: np.ndarray
    tpr: np.ndarray
    auc: float

    # Log loss
    log_loss: float

    # Per-split breakdown
    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics

    def summary(self) -> str:
        lines = [
            f"=== {self.model_name} ===",
            f"  Training time : {self.training_time_seconds:.2f}s",
            f"  AUC           : {self.auc:.6f}",
            f"  Log Loss      : {self.log_loss:.6f}",
            str(self.train_metrics),
            str(self.test_metrics),
        ]
        return "\n".join(lines)


@dataclass
class PreppedData:
    """Typed container for the four NumPy arrays produced by data preparation."""
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
