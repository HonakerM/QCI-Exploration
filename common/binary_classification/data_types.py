"""Define the shared data configuration and model result dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import json
from typing import Any
import numpy as np


# Recognized values for DataConfig.oversample_method.
OversampleMethod = Literal["smote", "random", "adasyn"]


@dataclass
class DataConfig:
    """Store the dataset and sampling settings used for a binary-classification run."""

    train_file: Optional[Path] = None
    test_file: Optional[Path] = None
    model_file: Optional[Path] = None

    model_name_override: str | None = None
    test_size: float = 0.3

    random_state: int = 42

    class_name: str = "Class"

    # -----------------------------------------------------------------
    # Legacy sample-size / balancing fields — behavior unchanged.
    # -----------------------------------------------------------------
    should_over_sample: bool = True
    enforce_equal_samples: bool = False
    over_sample_percentage: float = 1.0
    non_fraud_sample_size: int | None = None

    # -----------------------------------------------------------------
    # New, explicit sample-size / balancing fields. All default to "off"
    # (None) so existing configs are unaffected until you set one.
    # -----------------------------------------------------------------
    max_fraud_samples: int | None = None
    max_non_fraud_samples: int | None = None
    class_balance_ratio: float | None = None
    max_total_samples: int | None = None
    oversample_method: OversampleMethod = "smote"
    oversample_ratio: float | None = None

    # -----------------------------------------------------------------
    # Feature selection (unchanged).
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Data-quality / methodology fields. All default to "off" so existing
    # configs are unaffected.
    # -----------------------------------------------------------------
    preserve_natural_test_distribution: bool = False
    split_method: Literal["random", "chronological"] = "random"
    time_column: str = "Time"
    feature_scaling: Literal["none", "minmax", "standard"] = "none"
    drop_duplicates: bool = False
    log_transform_amount: bool = False
    amount_column: str = "Amount"

    # -----------------------------------------------------------------
    # Per-stage random states. Each falls back to `random_state` when
    # unset (None), so leaving them alone reproduces original behavior
    # exactly. Set them independently to isolate, e.g., "is this result
    # sensitive to the SMOTE seed" from "is it sensitive to the split
    # seed" in a DOE, rather than one seed confounding both.
    # -----------------------------------------------------------------
    sample_random_state: int | None = None
    split_random_state: int | None = None
    oversample_random_state: int | None = None

    def __post_init__(self) -> None:
        """Resolve legacy field aliases and validate the configuration values.

        Returns:
            None: Updates any unset modern fields from their legacy counterparts.
        """
        if self.max_non_fraud_samples is None:
            self.max_non_fraud_samples = self.non_fraud_sample_size

        if self.oversample_ratio is None:
            self.oversample_ratio = self.over_sample_percentage

        # Light validation — fail fast with a clear message rather than a
        # confusing downstream pandas/sklearn error.
        for name, value in (
            ("max_fraud_samples", self.max_fraud_samples),
            ("max_non_fraud_samples", self.max_non_fraud_samples),
            ("max_total_samples", self.max_total_samples),
        ):
            if value is not None and value <= 0:
                raise ValueError(
                    f"DataConfig.{name} must be a positive integer, got {value}"
                )

        if self.class_balance_ratio is not None and self.class_balance_ratio <= 0:
            raise ValueError(
                f"DataConfig.class_balance_ratio must be positive, got {self.class_balance_ratio}"
            )

        if self.oversample_ratio is not None and self.oversample_ratio <= 0:
            raise ValueError(
                f"DataConfig.oversample_ratio must be positive, got {self.oversample_ratio}"
            )

    @property
    def uses_class_balancing(self) -> bool:
        """Return whether the per-class balancing path should be used.

        Returns:
            bool: True when explicit balancing settings are active.
        """
        return (
            self.enforce_equal_samples
            or self.max_fraud_samples is not None
            or self.class_balance_ratio is not None
            or self.max_total_samples is not None
        )

    @property
    def effective_sample_random_state(self) -> int:
        """Return the random state to use for class sampling.

        Returns:
            int: Explicit sample seed or the default seed.
        """
        return (
            self.sample_random_state
            if self.sample_random_state is not None
            else self.random_state
        )

    @property
    def effective_split_random_state(self) -> int:
        """Return the random state to use for the train/test split.

        Returns:
            int: Explicit split seed or the default seed.
        """
        return (
            self.split_random_state
            if self.split_random_state is not None
            else self.random_state
        )

    @property
    def effective_oversample_random_state(self) -> int:
        """Return the random state to use for oversampling.

        Returns:
            int: Explicit oversampling seed or the default seed.
        """
        return (
            self.oversample_random_state
            if self.oversample_random_state is not None
            else self.random_state
        )

    @property
    def all_feature_names(self) -> list[str]:
        """Return the ordered list of model input feature names.

        Returns:
            list[str]: V-features, engineered features, and extra features combined.
        """
        return (
            self.v_feature_names
            + self.engineered_feature_names
            + self.additional_feature_names
        )


@dataclass
class DataSplit:
    """Store the train/test feature matrices and label arrays for one dataset split."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def __post_init__(self) -> None:
        """Verify that the training and test arrays have matching row counts.

        Returns:
            None: Raises an assertion if array dimensions do not match.
        """
        assert self.X_train.shape[0] == self.y_train.shape[0], (
            "X_train and y_train row counts must match"
        )
        assert self.X_test.shape[0] == self.y_test.shape[0], (
            "X_test and y_test row counts must match"
        )

    @property
    def n_features(self) -> int:
        """Return the number of feature columns in the training matrix.

        Returns:
            int: Feature count.
        """
        return self.X_train.shape[1]

    @property
    def n_train(self) -> int:
        """Return the number of rows in the training split.

        Returns:
            int: Training row count.
        """
        return self.X_train.shape[0]

    @property
    def n_test(self) -> int:
        """Return the number of rows in the test split.

        Returns:
            int: Test row count.
        """
        return self.X_test.shape[0]


@dataclass
class ClassificationMetrics:
    """Store the summary metrics for one classification split."""

    split: str
    precision: float
    recall: float
    f1: float
    accuracy: float
    confusion_matrix: np.ndarray

    def __str__(self) -> str:
        """Render a readable summary of the computed metrics.

        Returns:
            str: Formatted precision, recall, F1, accuracy, and confusion matrix.
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
class TimingInfo:
    """Store the timing for the main stages of a model run."""

    data_prep: float = 0.0
    fit: float | None = 0.0
    predict: float = 0.0
    adapter: dict[str, float] = field(default_factory=dict)

    @property
    def adapter_seconds(self) -> float:
        """Return the sum of all adapter-specific timing values.

        Returns:
            float: Total adapter runtime.
        """
        return sum(self.adapter.values())

    @property
    def total_seconds(self) -> float:
        """Return the total runtime across the recorded stages.

        Returns:
            float: Total seconds for data prep, fit, predict, and adapters.
        """
        fit_seconds = self.fit if self.fit is not None else self.adapter_seconds
        return self.data_prep + fit_seconds + self.predict

    def to_dict(self) -> dict[str, Any]:
        """Convert the timing data into a JSON-serializable dictionary.

        Returns:
            dict[str, Any]: Dictionary format suitable for JSON output.
        """
        data: dict[str, Any] = {
            "data_prep": self.data_prep,
            "fit": self.fit,
            "predict": self.predict,
        }
        if self.adapter:
            data["adapter"] = self.adapter
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimingInfo":
        """Rebuild a TimingInfo instance from its dictionary form.

        Args:
            data (dict[str, Any]): Serialized timing values.

        Returns:
            TimingInfo: Reconstructed timing object.
        """
        adapter_data = data.get("adapter", {})
        adapter = adapter_data if isinstance(adapter_data, dict) else {}
        return cls(
            data_prep=float(data.get("data_prep", 0.0)),
            fit=(None if data.get("fit") is None else float(data.get("fit", 0.0))),
            predict=float(data.get("predict", 0.0)),
            adapter={str(key): float(value) for key, value in adapter.items()},
        )


@dataclass
class ModelResults:
    """Store the full output from training and evaluating a single model."""

    model_name: str

    fpr: np.ndarray
    tpr: np.ndarray
    auc: float

    log_loss: float

    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics

    pr_precision: np.ndarray = field(default_factory=lambda: np.asarray([]))
    pr_recall: np.ndarray = field(default_factory=lambda: np.asarray([]))
    auc_pr: float = 0.0

    timing: TimingInfo = field(default_factory=TimingInfo)

    @property
    def training_time_seconds(self) -> float:
        """Return the total recorded run time for this model.

        Returns:
            float: Runtime in seconds.
        """
        return self.timing.total_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model output to a JSON-safe dictionary.

        Returns:
            dict[str, Any]: Dictionary form of the model result values.
        """
        return {
            "model_name": self.model_name,
            "training_time_seconds": self.training_time_seconds,
            "timing": self.timing.to_dict(),
            "fpr": self.fpr.tolist(),
            "tpr": self.tpr.tolist(),
            "auc": self.auc,
            "pr_precision": self.pr_precision.tolist(),
            "pr_recall": self.pr_recall.tolist(),
            "auc_pr": self.auc_pr,
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
        """Write the model result bundle to a JSON file.

        Args:
            path (str | Path): Destination file path.
            indent (int): Number of spaces to use in the JSON output.
        """
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelResults":
        """Rebuild a ModelResults instance from its dictionary form.

        Args:
            data (dict[str, Any]): Serialized model results.

        Returns:
            ModelResults: Reconstructed model result bundle.
        """
        timing_data = data.get("timing")
        if timing_data is None:
            timing = TimingInfo(fit=float(data.get("training_time_seconds", 0.0)))
        else:
            timing = TimingInfo.from_dict(timing_data)
        return cls(
            model_name=data["model_name"],
            timing=timing,
            fpr=np.asarray(data["fpr"], dtype=float),
            tpr=np.asarray(data["tpr"], dtype=float),
            auc=float(data["auc"]),
            pr_precision=np.asarray(data.get("pr_precision", []), dtype=float),
            pr_recall=np.asarray(data.get("pr_recall", []), dtype=float),
            auc_pr=float(data.get("auc_pr", 0.0)),
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
        """Load a saved model result bundle from a JSON file.

        Args:
            path (str | Path): Path to the result file.

        Returns:
            ModelResults: Reconstructed model result bundle.
        """
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def summary(self) -> str:
        """Render the full model summary as a human-readable string.

        Returns:
            str: Text summary with timing, AUC, log loss, and metrics.
        """
        return (
            f"=== {self.model_name} ===\n"
            f"  Training time : {self.training_time_seconds:.2f}s\n"
            f"  AUC           : {self.auc:.6f}\n"
            f"  Log Loss      : {self.log_loss:.6f}\n"
            f"{self.train_metrics}\n"
            f"{self.test_metrics}"
        )
