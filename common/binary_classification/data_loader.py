"""Load, clean, and split the fraud dataset used by the classifiers."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data_types import DataConfig
from ..logging import get_logger

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN

from .data_types import DataSplit


LOGGER = get_logger(__name__)


def get_data_split(cfg: DataConfig) -> DataSplit:
    """Loads and prepares data in one step.

    Args:
        cfg (DataConfig): Data configuration describing which files to load and how to
            split them.

    Returns:
        DataSplit: The train/test feature and label arrays.
    """
    df = load_data(cfg)
    return prep_data(df, cfg)


def load_data(cfg: DataConfig) -> pd.DataFrame:
    """Loads one or more CSV files, optionally concatenates them,
    engineers V_* features, and returns the combined DataFrame.

    Args:
        cfg (DataConfig): Data configuration specifying the train/test file paths and
            column names. Its v_feature_names field is populated with the
            discovered V-prefixed columns as a side effect.

    Returns:
        pd.DataFrame: The combined, feature-engineered DataFrame.
    """
    selected_paths: list[Path] = []
    if cfg.train_file is not None:
        selected_paths.append(Path(cfg.train_file))
    if cfg.test_file is not None:
        selected_paths.append(Path(cfg.test_file))

    if not selected_paths:
        raise ValueError("At least one of train_file or test_file must be provided")

    for path in selected_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}\n")

    frames: list[pd.DataFrame] = []
    if cfg.train_file is not None:
        LOGGER.info("Loading %s...", cfg.train_file)
        df_train = pd.read_csv(cfg.train_file)
        LOGGER.info("  Train rows: %s", f"{len(df_train):,}")
        frames.append(df_train)

    if cfg.test_file is not None:
        LOGGER.info("Loading %s...", cfg.test_file)
        df_test = pd.read_csv(cfg.test_file)
        LOGGER.info("  Test rows : %s", f"{len(df_test):,}")
        frames.append(df_test)

    if len(frames) == 1:
        df = frames[0].copy()
    else:
        df = pd.concat(frames, ignore_index=True)
        LOGGER.info("  Combined  : %s rows", f"{len(df):,}")

    df = _encode_class_name_to_val(df, cfg.class_name)
    df = _encode_categorical_to_v_fields(
        df,
        cfg.class_name,
        ignored_fields=cfg.additional_feature_names + [cfg.index_column],
    )

    cfg.v_feature_names = [str(col) for col in df if col.startswith("V")]

    if cfg.enable_engineered_features:
        df = _engineer_v_features(df, cfg.v_feature_names)
    LOGGER.info("Final dataset: %s rows x %s columns", f"{len(df):,}", len(df.columns))
    return df


def prep_data(df: pd.DataFrame, cfg: DataConfig) -> DataSplit:
    """Cleans, balances, label-encodes {-1, +1}, and splits into train/test arrays.

    Steps (default, `preserve_natural_test_distribution=False`):
        1. Optionally drop exact-duplicate rows (`cfg.drop_duplicates`) and
           log1p-transform Amount (`cfg.log_transform_amount`).
        2. Resolve the per-class row targets (see `_resolve_class_targets`)
           and sample fraud/non-fraud rows to match, OR fall back to the
           legacy dropna-only behavior — see `DataConfig.uses_class_balancing`
           for exactly which condition selects which path.
        3. If `cfg.max_total_samples` is set, apply a stratified trim so the
           resulting dataset never exceeds it, preserving the class ratio.
        4. Map the class column: 0 -> -1, 1 -> +1.
        5. Split into train/test (`cfg.split_method`: random-stratified or
           chronological).
        6. Optionally scale features (`cfg.feature_scaling`), fit on train
           only.
        7. If `cfg.should_over_sample`, oversample the training fold only,
           using `cfg.oversample_method` and `cfg.oversample_ratio`.

    If `cfg.preserve_natural_test_distribution` is True, the order changes:
    the split happens first (step 5), and steps 2-4 apply ONLY to the
    resulting training fold — the test fold is returned exactly as split,
    at its natural class distribution. See `_prep_data_natural_test`.

    Backward compatibility: when a DataConfig only sets the legacy fields
    (`enforce_equal_samples`, `non_fraud_sample_size`,
    `over_sample_percentage`) and leaves every new-style field at its
    default (including `preserve_natural_test_distribution=False`,
    `split_method="random"`, `feature_scaling="none"`,
    `drop_duplicates=False`, `log_transform_amount=False`), this function
    produces byte-identical output to the original implementation — same
    rows sampled in the same order, same split, same SMOTE call. See
    `DataConfig`'s docstring for the exact compatibility rules.

    Args:
        df (pd.DataFrame): Feature-engineered DataFrame, as returned by load_data().
        cfg (DataConfig): Data configuration describing the class column, sample size,
            test size, and feature columns to use.

    Returns:
        DataSplit: The resulting train/test feature and label arrays.
    """
    df = _apply_data_quality_steps(df, cfg)

    if cfg.preserve_natural_test_distribution:
        return _prep_data_natural_test(df, cfg)

    if cfg.uses_class_balancing:
        df = _sample_by_class(df, cfg)
    else:
        LOGGER.info(
            "enforce_equal_samples is False and no new-style balancing field is "
            "set -> skipping per-class sampling (legacy behavior: dropna only, "
            "natural class distribution kept)."
        )
        df = df.dropna().copy()

    df = _cap_total_samples(df, cfg)

    df[cfg.class_name] = df[cfg.class_name].map({0: -1, 1: 1})

    df_train, df_test = _split_dataframe(df, cfg)
    X_train = df_train[cfg.all_feature_names].to_numpy()
    y_train = df_train[cfg.class_name].to_numpy()
    X_test = df_test[cfg.all_feature_names].to_numpy()
    y_test = df_test[cfg.class_name].to_numpy()

    LOGGER.info(
        f"  Train: shape={X_train.shape}, label counts={dict(Counter(y_train))}"
    )
    LOGGER.info(f"  Test: shape={X_test.shape}, label counts={dict(Counter(y_test))}")

    X_train, X_test = _scale_features(X_train, X_test, cfg)

    if cfg.should_over_sample:
        X_train, y_train = _oversample(X_train, y_train, cfg)
        LOGGER.info(
            f"  Oversample Train: shape={X_train.shape}, label counts={dict(Counter(y_train))}"
        )

    return DataSplit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def _prep_data_natural_test(df: pd.DataFrame, cfg: DataConfig) -> DataSplit:
    """Implements the `preserve_natural_test_distribution=True` code path.

    Splits BEFORE any class-balancing, then applies balancing/limiting only
    to the training fold. The test fold is returned untouched at its
    natural class ratio — matching the paper's "test-fold remained
    original" protocol.

    Args:
        df (pd.DataFrame): The (already duplicate-dropped / log-transformed)
            dataset, still in its raw 0/1 class encoding.
        cfg (DataConfig): The data configuration driving every step.

    Returns:
        DataSplit: The resulting train/test feature and label arrays.
    """
    df = df.dropna().copy()

    df_train, df_test = _split_dataframe(df, cfg)
    LOGGER.info(
        "  preserve_natural_test_distribution: train=%s rows (label counts=%s), "
        "test=%s rows (label counts=%s, natural/untouched)",
        len(df_train),
        dict(Counter(df_train[cfg.class_name])),
        len(df_test),
        dict(Counter(df_test[cfg.class_name])),
    )

    if cfg.uses_class_balancing:
        df_train = _sample_by_class(df_train, cfg)
    df_train = _cap_total_samples(df_train, cfg)

    df_train[cfg.class_name] = df_train[cfg.class_name].map({0: -1, 1: 1})
    df_test[cfg.class_name] = df_test[cfg.class_name].map({0: -1, 1: 1})

    X_train = df_train[cfg.all_feature_names].to_numpy()
    y_train = df_train[cfg.class_name].to_numpy()
    X_test = df_test[cfg.all_feature_names].to_numpy()
    y_test = df_test[cfg.class_name].to_numpy()

    LOGGER.info(
        f"  Train (post-balancing): shape={X_train.shape}, label counts={dict(Counter(y_train))}"
    )
    LOGGER.info(
        f"  Test (natural): shape={X_test.shape}, label counts={dict(Counter(y_test))}"
    )

    X_train, X_test = _scale_features(X_train, X_test, cfg)

    if cfg.should_over_sample:
        X_train, y_train = _oversample(X_train, y_train, cfg)
        LOGGER.info(
            f"  Oversample Train: shape={X_train.shape}, label counts={dict(Counter(y_train))}"
        )

    return DataSplit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


# ---------------------------------------------------------------------------
# Data-quality / split / scaling helpers
# ---------------------------------------------------------------------------


def _apply_data_quality_steps(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Applies drop_duplicates and log_transform_amount, if enabled.

    No-op on both counts when the corresponding config fields are at their
    defaults (False) — keeps every prior config's behavior unchanged.

    Args:
        df (pd.DataFrame): The dataset to clean.
        cfg (DataConfig): The data configuration; uses `drop_duplicates`,
            `log_transform_amount`, and `amount_column`.

    Returns:
        pd.DataFrame: The (possibly) cleaned dataset.
    """
    if cfg.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates().copy()
        LOGGER.info(
            "  drop_duplicates: %s -> %s rows (removed %s exact duplicates)",
            before,
            len(df),
            before - len(df),
        )

    if cfg.log_transform_amount:
        if cfg.amount_column not in df.columns:
            LOGGER.info(
                "  log_transform_amount is True but column %r not found -> skipping",
                cfg.amount_column,
            )
        else:
            df = df.copy()
            df[cfg.amount_column] = np.log1p(df[cfg.amount_column].clip(lower=0))
            LOGGER.info("  Applied log1p transform to %s", cfg.amount_column)

    return df


def _split_dataframe(
    df: pd.DataFrame, cfg: DataConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a DataFrame into train/test folds per `cfg.split_method`.

    `split_method="random"` (default) performs a stratified random split —
    when `split_random_state` is unset, this uses `cfg.random_state` and
    produces the exact same row selection as the original
    `train_test_split(X, y, ...)` call on the equivalent arrays.

    `split_method="chronological"` sorts by `cfg.time_column` and takes the
    earliest `1 - test_size` fraction as train, the remainder as test — no
    shuffling, no stratification, so later rows never leak into training.

    Args:
        df (pd.DataFrame): The dataset to split. Its class column may be in
            either 0/1 or -1/+1 encoding; this function doesn't care.
        cfg (DataConfig): The data configuration; uses `split_method`,
            `time_column`, `test_size`, `class_name`, and
            `effective_split_random_state`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (df_train, df_test).
    """
    if cfg.split_method == "chronological":
        df_sorted = df.sort_values(cfg.time_column)
        n_test = int(round(len(df_sorted) * cfg.test_size))
        n_train = len(df_sorted) - n_test
        df_train = df_sorted.iloc[:n_train].copy()
        df_test = df_sorted.iloc[n_train:].copy()
        LOGGER.info(
            "  Chronological split by %s: train=%s rows (earliest), test=%s rows (latest)",
            cfg.time_column,
            len(df_train),
            len(df_test),
        )
        return df_train, df_test

    df_train, df_test = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.effective_split_random_state,
        stratify=df[cfg.class_name],
    )
    return df_train.copy(), df_test.copy()


def _scale_features(
    X_train: np.ndarray, X_test: np.ndarray, cfg: DataConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Scales features per `cfg.feature_scaling`, fit on the training fold only.

    `feature_scaling="none"` (default) returns the arrays unchanged, so
    every prior config is unaffected. Fitting only on `X_train` (never
    `X_test`) avoids leaking test-fold statistics into the transform.

    Args:
        X_train (np.ndarray): Training feature matrix.
        X_test (np.ndarray): Test feature matrix.
        cfg (DataConfig): The data configuration; uses `feature_scaling`.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X_train, X_test), scaled if requested.
    """
    if cfg.feature_scaling == "none":
        return X_train, X_test
    if cfg.feature_scaling == "minmax":
        scaler = MinMaxScaler()
    elif cfg.feature_scaling == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(
            f"Unknown feature_scaling {cfg.feature_scaling!r}; "
            "expected 'none', 'minmax', or 'standard'"
        )
    LOGGER.info("  Scaling features with %s (fit on train only)", cfg.feature_scaling)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _resolve_class_targets(
    fraud_total: int, non_fraud_total: int, cfg: DataConfig
) -> tuple[int, int]:
    """Determines how many fraud / non-fraud rows to sample.

    Priority order (see DataConfig's docstring "Field reference" section
    for the full rationale):
        1. `cfg.max_fraud_samples` for the fraud count, else all fraud rows.
        2. `cfg.class_balance_ratio` for the non-fraud count, computed as
           `round(class_balance_ratio * fraud_n)`.
        3. `cfg.max_non_fraud_samples` (aliased from `non_fraud_sample_size`
           in `__post_init__`) for the non-fraud count.
        4. If none of the above apply, non-fraud count defaults to the
           fraud count (a 1:1 balance) — this matches the original
           `enforce_equal_samples=True` default exactly.

    Both counts are clamped to the rows actually available.

    Args:
        fraud_total (int): Number of fraud rows present in the dataset.
        non_fraud_total (int): Number of non-fraud rows present in the dataset.
        cfg (DataConfig): The data configuration driving the decision.

    Returns:
        tuple[int, int]: (fraud_n, non_fraud_n) — the row counts to sample.
    """
    if cfg.max_fraud_samples is not None:
        fraud_n = min(cfg.max_fraud_samples, fraud_total)
        LOGGER.info(
            "  Fraud count source: max_fraud_samples=%s -> using %s of %s available",
            cfg.max_fraud_samples,
            fraud_n,
            fraud_total,
        )
    else:
        fraud_n = fraud_total
        LOGGER.info(
            "  Fraud count source: default -> keeping all %s fraud rows", fraud_n
        )

    if cfg.class_balance_ratio is not None:
        non_fraud_n = round(cfg.class_balance_ratio * fraud_n)
        LOGGER.info(
            "  Non-fraud count source: class_balance_ratio=%s x fraud_n(%s) -> %s",
            cfg.class_balance_ratio,
            fraud_n,
            non_fraud_n,
        )
    elif cfg.max_non_fraud_samples is not None:
        non_fraud_n = cfg.max_non_fraud_samples
        LOGGER.info(
            "  Non-fraud count source: max_non_fraud_samples/non_fraud_sample_size -> %s",
            non_fraud_n,
        )
    else:
        non_fraud_n = fraud_n
        LOGGER.info(
            "  Non-fraud count source: default (1:1 with fraud) -> %s", non_fraud_n
        )

    if non_fraud_n > non_fraud_total:
        LOGGER.info(
            "  Requested non-fraud count %s exceeds %s available -> clamping",
            non_fraud_n,
            non_fraud_total,
        )
        non_fraud_n = non_fraud_total

    return fraud_n, non_fraud_n


def _sample_by_class(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Samples fraud/non-fraud rows to the targets from `_resolve_class_targets`.

    When `cfg.max_fraud_samples` is unset (the default), fraud rows are
    kept exactly as-is with no `.sample()` call at all — matching the
    original implementation's behavior byte-for-byte (it never sub-sampled
    fraud, so it never touched row order via a random sample either).

    Args:
        df (pd.DataFrame): The full, feature-engineered dataset.
        cfg (DataConfig): The data configuration driving the sampling.

    Returns:
        pd.DataFrame: The concatenated, shuffled fraud + non-fraud subset.
    """
    df_fraud_all = df[df[cfg.class_name] == 1]
    df_non_fraud_all = df[df[cfg.class_name] == 0]

    fraud_n, non_fraud_n = _resolve_class_targets(
        fraud_total=len(df_fraud_all),
        non_fraud_total=len(df_non_fraud_all),
        cfg=cfg,
    )

    if cfg.max_fraud_samples is not None:
        df_fraud = df_fraud_all.sample(
            fraud_n, random_state=cfg.effective_sample_random_state
        )
    else:
        # Legacy path: keep every fraud row, untouched, no sampling call.
        df_fraud = df_fraud_all

    df_non_fraud = df_non_fraud_all.sample(
        non_fraud_n, random_state=cfg.effective_sample_random_state
    )

    result = (
        pd.concat([df_non_fraud, df_fraud])
        .sample(frac=1.0, random_state=cfg.effective_sample_random_state)
        .copy()
    )
    LOGGER.info(
        "  Class-balanced sample: %s fraud + %s non-fraud = %s rows",
        len(df_fraud),
        len(df_non_fraud),
        len(result),
    )
    return result


def _cap_total_samples(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Applies `cfg.max_total_samples`, if set, via a stratified trim.

    No-op (returns df unchanged) when `max_total_samples` is None or the
    dataset is already at or below the cap — this keeps every prior config
    (which never set this field) fully backward compatible.

    Args:
        df (pd.DataFrame): The dataset to potentially trim.
        cfg (DataConfig): The data configuration; only `max_total_samples`,
            `class_name`, and `random_state` are used.

    Returns:
        pd.DataFrame: The original df, or a stratified subsample of it.
    """
    if cfg.max_total_samples is None or len(df) <= cfg.max_total_samples:
        return df

    LOGGER.info(
        "  max_total_samples=%s < current %s rows -> taking a stratified subsample",
        cfg.max_total_samples,
        len(df),
    )
    trimmed, _ = train_test_split(
        df,
        train_size=cfg.max_total_samples,
        stratify=df[cfg.class_name],
        random_state=cfg.effective_sample_random_state,
    )
    LOGGER.info(
        "  After max_total_samples trim: %s rows, label counts=%s",
        len(trimmed),
        dict(Counter(trimmed[cfg.class_name])),
    )
    return trimmed


def _oversample(X_train, y_train, cfg: DataConfig):
    """Oversamples the training fold using `cfg.oversample_method`.

    `cfg.oversample_method="smote"` (the default) reproduces the original
    implementation exactly — same SMOTE call, same `sampling_strategy`
    value (via the `oversample_ratio`/`over_sample_percentage` alias).

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        cfg (DataConfig): The data configuration; uses `oversample_method`,
            `oversample_ratio`, and `random_state`.

    Returns:
        The resampled (X_train, y_train).
    """
    LOGGER.info(
        "  Oversampling with method=%s, ratio=%s",
        cfg.oversample_method,
        cfg.oversample_ratio,
    )
    if cfg.oversample_method == "smote":
        sampler = SMOTE(
            sampling_strategy=cfg.oversample_ratio,
            random_state=cfg.effective_oversample_random_state,
        )
    elif cfg.oversample_method == "random":
        sampler = RandomOverSampler(
            sampling_strategy=cfg.oversample_ratio,
            random_state=cfg.effective_oversample_random_state,
        )
    elif cfg.oversample_method == "adasyn":
        sampler = ADASYN(
            sampling_strategy=cfg.oversample_ratio,
            random_state=cfg.effective_oversample_random_state,
        )
    else:
        raise ValueError(
            f"Unknown oversample_method {cfg.oversample_method!r}; "
            "expected 'smote', 'random', or 'adasyn'"
        )
    return sampler.fit_resample(X_train, y_train)


# ---------------------------------------------------------------------------
# Private helpers (unchanged)
# ---------------------------------------------------------------------------


def _engineer_v_features(df: pd.DataFrame, v_cols: list[str]) -> pd.DataFrame:
    """Adds the eight row-wise aggregate features derived from the V columns.

    Args:
        df (pd.DataFrame): DataFrame containing the V-prefixed input columns.
        v_cols (list[str]): Names of the V-prefixed columns to aggregate over.

    Returns:
        pd.DataFrame: The DataFrame with Comp_* aggregate columns added (or unchanged if
        they are already present).
    """
    if "Comp_Sum" in df.columns:
        LOGGER.info("  Comp_* features already present, skipping engineering")
        return df

    LOGGER.info("Engineering Comp_* aggregate features...")
    v = df[v_cols].values
    df["Comp_Sum"] = v.sum(axis=1)
    df["Comp_Min"] = v.min(axis=1)
    df["Comp_Max"] = v.max(axis=1)
    df["Comp_Avg"] = v.mean(axis=1)
    df["Comp_Std"] = v.std(axis=1)
    df["Comp_Pos"] = (v > 0).sum(axis=1)
    df["Comp_Neg"] = (v < 0).sum(axis=1)
    df["Comp_Var"] = v.var(axis=1)
    LOGGER.info(
        "  Added Comp_Sum Comp_Min Comp_Max Comp_Avg Comp_Std Comp_Pos Comp_Neg Comp_Var"
    )
    return df


def _encode_categorical_to_v_fields(
    df: pd.DataFrame,
    class_col: str,
    smoothing: float = 10,
    ignored_fields: list[str] | None = None,
) -> pd.DataFrame:
    """Converts categorical columns into numerical values in [0, 1] using
    smoothed target (mean) encoding against class_col, for use with
    XGBoost.

    Each category is replaced by a smoothed average of the class label
    for rows with that category, then min-max scaled to [0, 1]. This
    keeps categories with similar target behavior numerically close,
    which gives XGBoost meaningful split points (unlike an arbitrary
    alphabetical/uniform encoding).

    NaNs are preserved as NaN (XGBoost handles missing values natively).

    Note: this fits the encoding on the full dataframe passed in, so if
    you have a separate test set, encode using stats from train only to
    avoid leakage.

    Args:
        df (pd.DataFrame): DataFrame containing the categorical columns to encode.
        class_col (str): Name of the target/label column used to compute the
            smoothed means.
        smoothing (float): Smoothing strength; higher values pull rare categories'
            encoded values closer to the global mean.
        ignored_fields (list[str] | None): Column names to exclude from encoding and renaming.

    Returns:
        pd.DataFrame: A copy of the DataFrame with categorical columns target-encoded
        and non-V feature columns renamed with a "V_" prefix.
    """
    # Setup ignored fields if not already
    if ignored_fields is None:
        ignored_fields = []

    df = df.copy()
    global_mean = df[class_col].mean()

    cat_cols = [
        col
        for col in df.columns
        if col != class_col
        and (
            df[col].dtype == "object"
            or isinstance(df[col].dtype, pd.CategoricalDtype)
            or df[col].dtype == "bool"
        )
        and (col not in ignored_fields)
    ]

    for col in cat_cols:
        stats = df.groupby(col)[class_col].agg(["mean", "count"])
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        mapping = smoothed.to_dict()

        encoded = df[col].map(mapping).astype(float)
        lo, hi = encoded.min(), encoded.max()
        if pd.notna(lo) and hi > lo:
            encoded = (encoded - lo) / (hi - lo)

        df[col] = encoded
        LOGGER.info(
            "  Encoded %s via target encoding -> [0, 1] (%d categories)",
            col,
            len(mapping),
        )

    feature_cols = [c for c in df.columns if c != class_col]
    rename_map = {}
    for col in feature_cols:
        if not col.startswith("V") and col not in ignored_fields:
            rename_map[col] = f"V_{col}"
    df = df.rename(columns=rename_map)

    return df


TRUE_VALUES = {"yes", "true", "1", "y", "t"}
FALSE_VALUES = {"no", "false", "0", "n", "f"}


def _encode_class_name_to_val(df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    """Encodes a boolean-like class column into numeric 0/1 values.

    Args:
        df (pd.DataFrame): DataFrame containing the class column.
        class_col (str): Name of the class column to encode.

    Returns:
        pd.DataFrame: The DataFrame with class_col encoded as 0/1 if it contained
        recognized boolean-like values, otherwise unchanged.
    """
    # Skip numeric columns
    if pd.api.types.is_numeric_dtype(df[class_col]):
        return df

    # Normalize values
    normalized = df[class_col].astype(str).str.strip().str.lower()

    # If every non-null value is a recognized boolean, convert it
    unique_values = set(normalized.dropna().unique())
    if unique_values.issubset(TRUE_VALUES | FALSE_VALUES):
        df[class_col] = normalized.map(lambda x: 1 if x in TRUE_VALUES else 0)

    LOGGER.info("  Encoded %s into numerical value", class_col)
    return df
