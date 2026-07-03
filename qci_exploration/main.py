"""
main.py
-------
Entry point for the Credit Card Fraud Detection pipeline.

Usage
-----
Classical models only (safe, no QPU cost):
    python main.py

Include quantum CVQBoost (⚠️ consumes Dirac-3 allocation):
    python main.py --enable-quantum

Save plots to disk instead of showing interactively:
    python main.py --save-plots

All options together:
    python main.py --enable-quantum --save-plots
"""

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

from config import CVQBoostConfig, DataConfig, XGBoostConfig
from data_loader import load_data
from data_prep import prep_data, xgboost_labels
from results import ModelResults
from visualization import plot_metric_comparison, plot_roc_curves


def _timestamp() -> str:
    return datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection: XGBoost vs CVQBoost"
    )
    parser.add_argument(
        "--enable-quantum",
        action="store_true",
        help=(
            "Enable CVQBoost training on QCi Dirac-3. "
            "⚠️  Consumes paid QPU allocation (~$0.22/run)."
        ),
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to PNG files instead of displaying interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv()  # Load QCI_TOKEN / QCI_API_URL from .env if present

    # ------------------------------------------------------------------
    # Configuration — all tunable settings live in config.py
    # ------------------------------------------------------------------
    data_cfg    = DataConfig()
    xgb_cfg     = XGBoostConfig()
    cvq_cfg     = CVQBoostConfig()

    overall_start = time.time()
    print(f"Pipeline start: {_timestamp()}")

    # ------------------------------------------------------------------
    # 1. Load & engineer features
    # ------------------------------------------------------------------
    df = load_data(data_cfg)

    # ------------------------------------------------------------------
    # 2. Balance & split   (labels → {-1, +1})
    # ------------------------------------------------------------------
    split = prep_data(df, data_cfg)
    print(
        f"\nData split ready: "
        f"{split.n_train} train rows, {split.n_test} test rows, "
        f"{split.n_features} features\n"
    )

    # ------------------------------------------------------------------
    # 3. XGBoost  (requires label re-scale to {0, 1})
    # ------------------------------------------------------------------
    from models.xgboost_model import train_and_evaluate as xgb_train
    xgb_split   = xgboost_labels(split)
    xgb_results = xgb_train(xgb_split, xgb_cfg)
    print()

    # ------------------------------------------------------------------
    # 4. CVQBoost  (optional — gated by --enable-quantum)
    # ------------------------------------------------------------------
    all_results: list[ModelResults] = [xgb_results]

    if args.enable_quantum:
        from models.cvqboost_model import (
            check_credentials,
            train_and_evaluate as cvq_train,
        )
        print("⚠️  Quantum execution enabled — submitting to QCi Dirac-3...")
        check_credentials(cvq_cfg)
        # CVQBoost uses the original {-1, +1} labels
        cvq_results = cvq_train(split, cvq_cfg)
        all_results.append(cvq_results)
        print()
    else:
        print(
            "CVQBoost skipped (pass --enable-quantum to include it).\n"
            "Remember: each run consumes ~1 QPU second on Dirac-3.\n"
        )

    # ------------------------------------------------------------------
    # 5. Visualize
    # ------------------------------------------------------------------
    roc_path    = Path("roc_curves.png")    if args.save_plots else None
    metric_path = Path("metric_comparison.png") if args.save_plots else None

    plot_roc_curves(all_results, save_path=roc_path)
    plot_metric_comparison(all_results, save_path=metric_path)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - overall_start
    print(f"Overall execution time: {elapsed:.2f}s")
    print(f"Pipeline end: {_timestamp()}")


if __name__ == "__main__":
    main()
