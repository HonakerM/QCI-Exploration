"""
common
------
Shared helpers for the credit card fraud detection scripts.

Public API
----------
Config dataclasses:
    DataConfig

Result dataclasses:
    DataSplit, ModelResults, ClassificationMetrics

Data pipeline:
    load_data, prep_data

Evaluation:
    compute_metrics, print_results

Visualization:
    plot_roc_curves, plot_metric_comparison
"""

from .data_types import DataConfig, ClassificationMetrics, DataSplit, ModelResults
from .data_loader import load_data
from .data_prep import prep_data
from .evaluation import compute_metrics, print_results
from .visualization import plot_metric_comparison, plot_roc_curves

__all__ = [
    # config
    "DataConfig",
    # results
    "DataSplit",
    "ModelResults",
    "ClassificationMetrics",
    # data
    "load_data",
    "prep_data",
    # evaluation
    "compute_metrics",
    "print_results",
    # visualization
    "plot_roc_curves",
    "plot_metric_comparison",
]
