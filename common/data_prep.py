"""
common/data_prep.py
-------------------
Balances the dataset and produces typed train/test splits.

Label convention
----------------
prep_data() maps the raw Kaggle Class column {0, 1} to {-1, +1}:

    Class 0 (non-fraud)  →  -1
    Class 1 (fraud)      →  +1

Individual scripts are responsible for any further label transformation
their model requires before calling fit().
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_types import DataConfig, DataSplit


def prep_data(df: pd.DataFrame, cfg: DataConfig) -> DataSplit:
    """
    Balance, label-encode {-1, +1}, and split into train/test arrays.

    Steps
    -----
    1. Undersample non-fraud rows to ``cfg.non_fraud_sample_size``.
    2. Concatenate with all fraud rows, then shuffle.
    3. Map Class column: 0 → -1, 1 → +1.
    4. Build feature matrix from ``cfg.all_feature_names``.
    5. Stratified train/test split.
    """
    df_non_fraud = df[df["Class"] == 0].sample(
        cfg.non_fraud_sample_size, random_state=cfg.random_state
    )
    df_fraud = df[df["Class"] == 1]

    balanced = (
        pd.concat([df_non_fraud, df_fraud])
        .sample(frac=1.0, random_state=cfg.random_state)
        .copy()
    )
    balanced["Class"] = balanced["Class"].map({0: -1, 1: 1})

    X = balanced[cfg.all_feature_names].to_numpy()
    y = balanced["Class"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    _log_split("Train", X_train, y_train)
    _log_split("Test ", X_test, y_test)

    return DataSplit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _log_split(name: str, X: np.ndarray, y: np.ndarray) -> None:
    print(f"  {name}: shape={X.shape}, label counts={dict(Counter(y))}")
