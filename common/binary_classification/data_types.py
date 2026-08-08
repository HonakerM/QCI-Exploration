"""Dataclasses shared by the binary classification training scripts."""

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
    """Paths and split settings shared by both scripts.

    This config has two layers of fields:

    - **Legacy fields** (`non_fraud_sample_size`, `enforce_equal_samples`,
      `over_sample_percentage`) behave EXACTLY as they always have. Any
      existing yaml that only sets these will produce byte-identical
      sampling to before — nothing about their behavior changed.
    - **New, explicit fields** (`max_fraud_samples`, `max_non_fraud_samples`,
      `class_balance_ratio`, `max_total_samples`, `oversample_ratio`,
      `oversample_method`) give more direct, individually-named control
      over each sampling decision. They're `None`/off by default and only
      change behavior when you actually set them.

    Where a new field and a legacy field describe the same thing
    (`max_non_fraud_samples` vs. `non_fraud_sample_size`, `oversample_ratio`
    vs. `over_sample_percentage`), setting either one works — see
    `__post_init__` for exactly how they're reconciled, and the "Field
    reference" section below for which one wins if both are set.

    Field reference
    ---------------
    Sample-count controls (evaluated in this priority order):
        1. `class_balance_ratio` — if set, non-fraud count is computed as
           `round(class_balance_ratio * fraud_count)`, overriding
           `max_non_fraud_samples`/`non_fraud_sample_size`.
        2. `max_non_fraud_samples` (or its legacy alias
           `non_fraud_sample_size`) — an explicit non-fraud row count.
        3. If neither is set: non-fraud count defaults to the fraud count
           (i.e. a 1:1 class balance) — this is the original default
           behavior of `enforce_equal_samples=True`.

    Fraud count is controlled independently by `max_fraud_samples`. Leaving
    it `None` (the default) keeps ALL fraud rows, matching original
    behavior, where fraud was never sub-sampled.

    `max_total_samples` is an overall cap applied AFTER the above
    class-count logic. If the resulting dataset is larger than the cap, a
    stratified subsample (preserving the class ratio) is taken.

    Important compatibility note: exactly like the original code, per-class
    sampling only happens when `enforce_equal_samples=True` OR you've set
    one of the new-style balancing fields (`max_fraud_samples`,
    `class_balance_ratio`, or `max_total_samples`). Setting only
    `max_non_fraud_samples`/`non_fraud_sample_size` with
    `enforce_equal_samples=False` has NO effect, matching the original
    code, which ignored `non_fraud_sample_size` entirely in that branch.
    If you want an explicit non-fraud cap to actually apply, either set
    `enforce_equal_samples=True` or set `class_balance_ratio` /
    `max_total_samples` alongside it.

    Attributes:
        train_file (Optional[Path]): Path to the training CSV file, if any.
        test_file (Optional[Path]): Path to the test CSV file, if any.
        test_size (float): Fraction of the balanced dataset held out for evaluation.
        non_fraud_sample_size (int | None): [LEGACY — see max_non_fraud_samples]
            Number of non-fraud rows sampled to balance the dataset (fraud
            rows are kept in full unless max_fraud_samples is set). Still
            fully supported; new configs should prefer
            `max_non_fraud_samples` for clarity.
        max_fraud_samples (int | None): Caps the number of fraud rows used,
            sampled randomly with `random_state`. `None` (default) keeps
            every fraud row — this matches all original behavior, which
            never sub-sampled the fraud class.
        max_non_fraud_samples (int | None): Explicit, clearly-named cap on
            non-fraud rows. Functionally identical to
            `non_fraud_sample_size`; if both are set, this field wins. If
            neither is set, `__post_init__` copies
            `non_fraud_sample_size` into this field so the rest of the
            code only has to look in one place.
        class_balance_ratio (float | None): Desired ratio of
            non-fraud : fraud rows in the pre-split, pre-oversample
            dataset. E.g. `1.0` means equal counts (the original
            `enforce_equal_samples=True` default), `5.0` means five
            non-fraud rows for every fraud row. Takes priority over
            `max_non_fraud_samples`/`non_fraud_sample_size` when set.
        max_total_samples (int | None): Hard cap on the total row count
            after class-balancing. If the balanced dataset exceeds this, a
            stratified subsample preserving the current class ratio is
            taken. `None` (default) means no cap — matches original
            behavior.
        random_state (int): Seed used for sampling and the train/test split.
        class_name (str): Name of the target/label column.
        v_feature_names (list[str]): Names of the PCA-transformed input columns.
            Populated automatically by `load_data()` from whichever columns
            start with "V" — any value set here is overwritten and has no
            effect on which columns are used.
        engineered_feature_names (list[str]): Names of the aggregate features added on
            top of the V-prefixed columns.
        additional_feature_names (list[str]): Names of any extra raw feature columns
            to include alongside the V and engineered features. Also used
            as the `ignored_fields` list for the categorical target-encoder
            in `load_data()` — columns named here are never renamed/encoded
            as categorical features. Has no effect on datasets (like
            mlg-ulb) with no categorical columns to encode.
        index_column (str): Name of the row identifier column.
        should_over_sample (bool): Whether to apply oversampling to the
            training fold after the split. Oversampling is never applied to
            the test fold.
        oversample_method (Literal["smote", "random", "adasyn"]): Which
            imbalanced-learn oversampler to use. Defaults to `"smote"`,
            matching original behavior exactly. `"random"` uses
            `RandomOverSampler` (duplicates existing minority rows —
            cheaper, no synthetic interpolation). `"adasyn"` uses ADASYN
            (like SMOTE, but focuses synthetic samples on harder-to-learn
            minority points).
        over_sample_percentage (float): [LEGACY — see oversample_ratio]
            Passed as `sampling_strategy` to the oversampler. Still fully
            supported; new configs should prefer `oversample_ratio` for
            clarity.
        oversample_ratio (float | None): Explicit, clearly-named version of
            `over_sample_percentage` — the desired minority:majority row
            ratio in the training fold AFTER oversampling (e.g. `1.0` =
            fully balanced classes, `0.5` = minority ends up at half the
            majority's count). If unset, `__post_init__` copies
            `over_sample_percentage` into this field.
        preserve_natural_test_distribution (bool): If True, the train/test
            split happens BEFORE any class-balancing/sampling, and only the
            training fold is balanced/limited/oversampled — the test fold
            is returned exactly as split, at its natural class ratio. This
            matches the paper's evaluation protocol ("test-fold remained
            original"). If False (default), matches the original
            implementation: the whole dataset is balanced first, so both
            train and test come from the same balanced pool.
        split_method (Literal["random", "chronological"]): `"random"`
            (default) performs the original stratified random split.
            `"chronological"` sorts by `time_column` and takes the earliest
            `1 - test_size` fraction as train, the rest as test — tests
            robustness to distribution shift instead of leaking
            near-future rows into training.
        time_column (str): Column used to sort rows when
            `split_method="chronological"`. Defaults to `"Time"`, matching
            mlg-ulb.
        feature_scaling (Literal["none", "minmax", "standard"]): `"none"`
            (default) applies no scaling, matching original behavior.
            `"minmax"` scales each feature to [0, 1] (matching the paper's
            preprocessing) and `"standard"` z-scores each feature. Fit on
            the training fold only, then applied to both folds, to avoid
            leakage.
        drop_duplicates (bool): If True, exact-duplicate rows are dropped
            before any sampling/splitting. mlg-ulb is known to contain
            ~1,081 exact duplicate rows; left in, a duplicate can land in
            both the train and test folds, which is direct leakage.
            Defaults to False, matching original behavior.
        log_transform_amount (bool): If True, applies `log1p` to
            `amount_column` before feature extraction — `Amount` is
            heavily right-skewed, and this is a standard fix that
            particularly helps distance/linear-based classifiers (KNN,
            LDA, logistic regression). Defaults to False.
        amount_column (str): Column `log_transform_amount` applies to.
            Defaults to `"Amount"`.
        sample_random_state (int | None): Random state for the
            fraud/non-fraud `.sample()` calls. Falls back to
            `random_state` when `None` (default) — matches original
            behavior.
        split_random_state (int | None): Random state for the train/test
            split. Falls back to `random_state` when `None` (default) —
            matches original behavior.
        oversample_random_state (int | None): Random state for the
            oversampler. Falls back to `random_state` when `None`
            (default) — matches original behavior.
    """

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
        """Reconciles legacy fields with their new, explicitly-named aliases.

        This runs once, right after the dataclass is constructed (whether
        via `DataConfig(...)` directly or `DataConfig(**yaml_dict)`). It
        never overwrites a value you explicitly set on the new-style
        field — it only fills the new field in when you left it at its
        default `None`, using whatever the legacy field holds. This is
        what makes old yaml configs (which only ever set the legacy
        fields) produce identical behavior to before.

        Returns:
            None.
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
                raise ValueError(f"DataConfig.{name} must be a positive integer, got {value}")

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
        """Whether prep_data() will run the per-class sampling branch.

        Mirrors the exact condition used in `prep_data()`: True if
        `enforce_equal_samples` is set, or if any new-style balancing field
        (`max_fraud_samples`, `class_balance_ratio`, `max_total_samples`)
        is set. Note `max_non_fraud_samples`/`non_fraud_sample_size` alone
        does NOT trigger this branch — see the class docstring's
        "Important compatibility note".

        Returns:
            bool: True if class-balanced sampling will run, False if the
            legacy dropna-only fallback will run instead.
        """
        return (
            self.enforce_equal_samples
            or self.max_fraud_samples is not None
            or self.class_balance_ratio is not None
            or self.max_total_samples is not None
        )

    @property
    def effective_sample_random_state(self) -> int:
        """Returns sample_random_state, falling back to random_state."""
        return self.sample_random_state if self.sample_random_state is not None else self.random_state

    @property
    def effective_split_random_state(self) -> int:
        """Returns split_random_state, falling back to random_state."""
        return self.split_random_state if self.split_random_state is not None else self.random_state

    @property
    def effective_oversample_random_state(self) -> int:
        """Returns oversample_random_state, falling back to random_state."""
        return (
            self.oversample_random_state
            if self.oversample_random_state is not None
            else self.random_state
        )

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
class TimingInfo:
    """Durations for the main stages of a model run.

    Attributes:
        data_prep (float): Time spent preparing the dataset.
        fit (float | None): Time spent fitting the model, if captured directly.
        predict (float): Time spent generating predictions.
        adapter (dict[str, float]): Adapter-specific timing measurements.
    """

    data_prep: float = 0.0
    fit: float | None = 0.0
    predict: float = 0.0
    adapter: dict[str, float] = field(default_factory=dict)

    @property
    def adapter_seconds(self) -> float:
        """Returns the total adapter-specific runtime in seconds."""
        return sum(self.adapter.values())

    @property
    def total_seconds(self) -> float:
        """Returns the total recorded runtime in seconds."""
        fit_seconds = self.fit if self.fit is not None else self.adapter_seconds
        return self.data_prep + fit_seconds + self.predict

    def to_dict(self) -> dict[str, Any]:
        """Converts this timing info into a JSON-serializable dictionary."""
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
        """Reconstructs timing info from a dictionary representation."""
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
    """Everything produced by training and evaluating one model.

    Attributes:
        model_name (str): Name of the trained model.
        timing (TimingInfo): Stage timings for the run.
        fpr (np.ndarray): False positive rates for the ROC curve.
        tpr (np.ndarray): True positive rates for the ROC curve.
        auc (float): Area under the ROC curve.
        log_loss (float): Log loss on the test split.
        train_metrics (ClassificationMetrics): Classification metrics computed on the train split.
        test_metrics (ClassificationMetrics): Classification metrics computed on the test split.
    """

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
        """Returns the total recorded runtime in seconds."""
        return self.timing.total_seconds

    def to_dict(self) -> dict[str, Any]:
        """Converts these results into a JSON-serializable dictionary.

        Returns:
            dict[str, Any]: A dictionary representation of all fields, with NumPy arrays
            converted to lists.
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