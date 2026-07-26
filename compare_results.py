"""CLI for comparing saved model results with ROC and metric plots."""

import csv
from pathlib import Path

import typer

from common.binary_classification.data_types import ModelResults
from common.binary_classification.evaluation import print_results
from common.binary_classification.visualization import (
    plot_metric_comparison,
    plot_roc_curves,
    plot_timing_comparison,
)

app = typer.Typer()


def save_results_csv(results: list[ModelResults], path: Path) -> None:
    """Writes one row per model's test results to a CSV file.

    Each row includes the model name, AUC, log loss, test classification
    metrics, and timing breakdown. Timing is split into one column per
    stage (data_prep, fit, predict) plus one column per adapter-specific
    timing label. Since adapter timing labels can differ between results,
    the column set is the union of all labels seen across the given
    results; any result missing a given label gets a blank cell for it.

    Args:
        results (list[ModelResults]): Model results to write.
        path (Path): Destination CSV file path.
    """
    base_fields = [
        "model_name",
        "auc",
        "log_loss",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_accuracy",
        "timing_data_prep",
        "timing_fit",
        "timing_predict",
        "timing_total_seconds",
    ]

    # Union of all adapter timing labels across results, in first-seen order.
    adapter_labels: list[str] = []
    for r in results:
        for label in r.timing.adapter:
            if label not in adapter_labels:
                adapter_labels.append(label)

    adapter_fields = [f"timing_{label}" for label in adapter_labels]
    fieldnames = base_fields + adapter_fields

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row: dict[str, object] = {
                "model_name": r.model_name,
                "auc": r.auc,
                "log_loss": r.log_loss,
                "test_precision": r.test_metrics.precision,
                "test_recall": r.test_metrics.recall,
                "test_f1": r.test_metrics.f1,
                "test_accuracy": r.test_metrics.accuracy,
                "timing_data_prep": r.timing.data_prep,
                "timing_fit": r.timing.fit,
                "timing_predict": r.timing.predict,
                "timing_total_seconds": r.timing.total_seconds,
            }
            for label in adapter_labels:
                # Create a column name for each adapter timing label.
                col = f"timing_{label}"
                # Leave blank if this result doesn't have this adapter label.
                row[col] = r.timing.adapter.get(label, "")

            writer.writerow(row)


@app.command()
def main(
    results_files: list[Path] = typer.Argument(
        ...,
        help="Paths to saved ModelResults JSON files or folders containing JSON files",
    ),
    save_file: Path | None = None,
    csv_file: Path | None = None,
    display: bool = True,
):
    """Loads results JSONs and plots ROC curves, metrics, and timing comparisons.

    Args:
        results_files (list[Path]): Paths to saved ModelResults JSON files or folders
            containing JSON files.
        save_file (Path): If provided, save the comparison results to this file.
        csv_file (Path): If provided, save a CSV summary (AUC, log loss, test
            metrics, and per-stage timing) with one row per model result.
    """
    resolved_files: list[Path] = []
    for path in results_files:
        if path.is_dir():
            matches = sorted(path.rglob("*.json"))
            if matches:
                resolved_files.extend(matches)
        else:
            resolved_files.append(path)

    results = [ModelResults.load(f) for f in resolved_files]

    for r in results:
        print_results(r)

    if csv_file:
        save_results_csv(results, csv_file)

    roc_path = None
    metric_path = None
    timing_path = None
    if save_file:
        save_file.parent.mkdir(parents=True, exist_ok=True)
        roc_path = save_file.parent / (save_file.stem + "_roc.png")
        metric_path = save_file.parent / (save_file.stem + "_metrics.png")
        timing_path = save_file.parent / (save_file.stem + "_timing.png")

    if display or save_file:
        plot_roc_curves(results, save_path=roc_path)
        plot_metric_comparison(results, save_path=metric_path)
        plot_timing_comparison(results, save_path=timing_path)


if __name__ == "__main__":
    app()
