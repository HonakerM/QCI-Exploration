from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from qci_exploration.data_types import DataConfig, PreppedData
import pandas as pd


def _timestamp() -> str:
    return datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")


def get_data(data_config: DataConfig)->PreppedData:
    df = load_data(data_config.train_path, data_config.test_path)
    return prep_data(df, data_config)

def load_data(train_file: Path, test_file: Path) -> pd.DataFrame:
    """
    Load train.csv and test.csv, combine them, and return a single
    feature-engineered DataFrame ready for prep_data().

    Parameters
    ----------
    train_file : Path
        Path to the training CSV file.
    test_file : Path
        Path to the testing CSV file.

    Returns
    -------
    pd.DataFrame
        Combined dataset with engineered V_* columns added.

    Raises
    ------
    FileNotFoundError
        If either CSV path does not exist.
    """
    print(_timestamp())

    for path in (train_file, test_file):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Download train.csv and test.csv from:\n"
                "https://www.kaggle.com/competitions/playground-series-s3e4/data"
            )

    print(f"Loading {train_file}...")
    df_train = pd.read_csv(train_file)
    print(f"  Train rows: {len(df_train):,}")

    print(f"Loading {test_file}...")
    df_test = pd.read_csv(test_file)
    print(f"  Test rows : {len(df_test):,}")

    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"  Combined  : {len(df):,} rows")
    
    # Engineer V_Sum, V_Min, V_Max, V_Avg, V_Std, V_Pos, V_Neg, V_Var if not present
    v_cols = [f"V{i}" for i in range(1, 29)]
    if "V_Sum" not in df.columns:
        print("Engineering V_* features...")
        v_data = df[v_cols].values
        df["V_Sum"] = v_data.sum(axis=1)
        df["V_Min"] = v_data.min(axis=1)
        df["V_Max"] = v_data.max(axis=1)
        df["V_Avg"] = v_data.mean(axis=1)
        df["V_Std"] = v_data.std(axis=1)
        df["V_Pos"] = (v_data > 0).sum(axis=1)
        df["V_Neg"] = (v_data < 0).sum(axis=1)
        df["V_Var"] = v_data.var(axis=1)
        print("  ✓ Engineered 8 features")
    else:
        print("  ✓ V_* features already present")

    print(f"Final dataset: {len(df):,} rows, {len(df.columns)} features")

    print(datetime.now().strftime("%m/%d/%Y %I:%M:%S %p"))

    print(f"Final dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(_timestamp())
    return df



def prep_data(df: pd.DataFrame, cfg: DataConfig) -> PreppedData:
    df_non_fraud = df[df["Class"] == 0].sample(
        cfg.non_fraud_sample_size, random_state=cfg.random_state
    )
    df_fraud = df[df["Class"] == 1]

    balanced = pd.concat([df_non_fraud, df_fraud]).sample(frac=1.0, random_state=cfg.random_state)
    balanced = balanced.copy()
    balanced["Class"] = balanced["Class"].apply(lambda x: -1 if x == 0 else 1)

    X = balanced[cfg.all_feature_names].to_numpy()
    y = balanced["Class"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    return PreppedData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )





def validate_qboost_labels(split: PreppedData) -> None:
    """
    Assert that labels are exactly {-1, +1} as required by QBoostClassifier.

    Raises
    ------
    AssertionError if unexpected label values are found.
    """
    unexpected_test = split.y_test[~np.isin(split.y_test, [-1, 1])]
    if len(unexpected_test):
        raise AssertionError(
            f"QBoost requires labels in {{-1, 1}}.  "
            f"Found unexpected values: {unexpected_test}"
        )
