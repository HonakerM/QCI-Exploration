"""Loads raw fraud-detection CSVs, combines them, and engineers features."""

from pathlib import Path

import pandas as pd

from .data_types import DataConfig
from ..logging import get_logger

from collections import Counter

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

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

    df = _engineer_v_features(df, cfg.v_feature_names)
    LOGGER.info("Final dataset: %s rows x %s columns", f"{len(df):,}", len(df.columns))
    return df


def prep_data(df: pd.DataFrame, cfg: DataConfig) -> DataSplit:
    """Balances, label-encodes {-1, +1}, and splits into train/test arrays.

    Steps:
        1. Undersample non-fraud rows to cfg.non_fraud_sample_size.
        2. Concatenate with all fraud rows, then shuffle.
        3. Map the class column: 0 -> -1, 1 -> +1.
        4. Build the feature matrix from cfg.all_feature_names.
        5. Perform a stratified train/test split.

    Args:
        df (pd.DataFrame): Feature-engineered DataFrame, as returned by load_data().
        cfg (DataConfig): Data configuration describing the class column, sample size,
            test size, and feature columns to use.

    Returns:
        DataSplit: The resulting train/test feature and label arrays.
    """
    if cfg.limit_sample_size:
        df_non_fraud = df[df[cfg.class_name] == 0].sample(
            cfg.non_fraud_sample_size, random_state=cfg.random_state
        )
        df_fraud = df[df[cfg.class_name] == 1]

        df = (
            pd.concat([df_non_fraud, df_fraud])
            .sample(frac=1.0, random_state=cfg.random_state)
            .copy()
        )
    else:
        df = df.dropna().copy()

    df[cfg.class_name] = df[cfg.class_name].map({0: -1, 1: 1})

    X = df[cfg.all_feature_names].to_numpy()
    y = df[cfg.class_name].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    LOGGER.info(
        f"  Train: shape={X_train.shape}, label counts={dict(Counter(y_train))}"
    )
    LOGGER.info(f"  Test: shape={X_test.shape}, label counts={dict(Counter(y_test))}")

    if cfg.should_over_sample:
        smote = SMOTE(random_state=cfg.random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        LOGGER.info(
            f"  Oversample Train: shape={X_train.shape}, label counts={dict(Counter(y_train))}"
        )

    return DataSplit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


# ---------------------------------------------------------------------------
# Private helpers
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
