"""
common/data_loader.py
---------------------
Loads the raw Kaggle CSVs, combines them, and engineers the eight
aggregate V_* features.  Returns a single analysis-ready DataFrame.
"""

from pathlib import Path

import pandas as pd

from .data_types import DataConfig
from .logging import get_logger


LOGGER = get_logger(__name__)


def load_data(cfg: DataConfig) -> pd.DataFrame:
    """
    Load one or more CSV files, optionally concatenate them,
    engineer V_* features, and return the combined DataFrame.

    Raises
    ------
    FileNotFoundError
        If any selected CSV is missing — includes the Kaggle download URL.
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
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Download train.csv and test.csv from:\n"
                "https://www.kaggle.com/competitions/playground-series-s3e4/data"
            )

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

    df = _engineer_v_features(df, cfg.v_feature_names)
    LOGGER.info("Final dataset: %s rows x %s columns", f"{len(df):,}", len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _engineer_v_features(df: pd.DataFrame, v_cols: list[str]) -> pd.DataFrame:
    """Add the eight row-wise aggregate features derived from V1-V28."""
    if "V_Sum" in df.columns:
        LOGGER.info("  V_* features already present, skipping engineering")
        return df

    LOGGER.info("Engineering V_* aggregate features...")
    v = df[v_cols].values
    df["V_Sum"] = v.sum(axis=1)
    df["V_Min"] = v.min(axis=1)
    df["V_Max"] = v.max(axis=1)
    df["V_Avg"] = v.mean(axis=1)
    df["V_Std"] = v.std(axis=1)
    df["V_Pos"] = (v > 0).sum(axis=1)
    df["V_Neg"] = (v < 0).sum(axis=1)
    df["V_Var"] = v.var(axis=1)
    LOGGER.info("  Added V_Sum V_Min V_Max V_Avg V_Std V_Pos V_Neg V_Var")
    return df
