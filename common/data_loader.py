"""
common/data_loader.py
---------------------
Loads the raw Kaggle CSVs, combines them, and engineers the eight
aggregate V_* features.  Returns a single analysis-ready DataFrame.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from .data_types import DataConfig


def load_data(cfg: DataConfig) -> pd.DataFrame:
    """
    Load train.csv + test.csv, concatenate them, engineer V_* features,
    and return the combined DataFrame.

    Raises
    ------
    FileNotFoundError
        If either CSV is missing — includes the Kaggle download URL.
    """
    print(_ts())

    for path in (cfg.train_file, cfg.test_file):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Download train.csv and test.csv from:\n"
                "https://www.kaggle.com/competitions/playground-series-s3e4/data"
            )

    print(f"Loading {cfg.train_file}...")
    df_train = pd.read_csv(cfg.train_file)
    print(f"  Train rows: {len(df_train):,}")

    print(f"Loading {cfg.test_file}...")
    df_test = pd.read_csv(cfg.test_file)
    print(f"  Test rows : {len(df_test):,}")

    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"  Combined  : {len(df):,} rows")

    df = _engineer_v_features(df, cfg.v_feature_names)
    print(f"Final dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(_ts())
    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _engineer_v_features(df: pd.DataFrame, v_cols: list[str]) -> pd.DataFrame:
    """Add the eight row-wise aggregate features derived from V1-V28."""
    if "V_Sum" in df.columns:
        print("  ✓ V_* features already present, skipping engineering")
        return df

    print("Engineering V_* aggregate features...")
    v = df[v_cols].values
    df["V_Sum"] = v.sum(axis=1)
    df["V_Min"] = v.min(axis=1)
    df["V_Max"] = v.max(axis=1)
    df["V_Avg"] = v.mean(axis=1)
    df["V_Std"] = v.std(axis=1)
    df["V_Pos"] = (v > 0).sum(axis=1)
    df["V_Neg"] = (v < 0).sum(axis=1)
    df["V_Var"] = v.var(axis=1)
    print("  ✓ V_Sum  V_Min  V_Max  V_Avg  V_Std  V_Pos  V_Neg  V_Var")
    return df


def _ts() -> str:
    return datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")
