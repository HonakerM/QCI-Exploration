"""Compare saved model runs and render summary plots."""

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
    """Write the saved model metrics and timings to a CSV summary file.

    Args:
        results (list[ModelResults]): Results to export.
        path (Path): Destination CSV path.
    """
    base_fields = [
        "model_name",
        "run_label",
        "random_state",
        "auc_roc",
        "auc_pr",
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
                "run_label": r.run_label,
                "random_state": r.random_state,
                "auc_roc": r.auc,
                "auc_pr": getattr(r, "auc_pr", 0.0),
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
    display: bool = True,
):
    """Load saved results and render the comparison plots.

    Args:
        results_files (list[Path]): JSON files or directories to compare.
        save_file (Path | None): Optional base name for exported plot files.
        display (bool): Whether to show the generated plots in a window.
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

    if save_file:
        csv_file = save_file.parent / (save_file.stem + ".csv")
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
        try:
            from common.binary_classification.visualization import plot_pr_curves

            plot_pr_curves(
                results,
                save_path=save_file.parent / (save_file.stem + "_pr.png")
                if save_file
                else None,
            )
        except Exception:
            # Optional PR plotting; ignore if unavailable
            pass
        plot_metric_comparison(results, save_path=metric_path)
        plot_timing_comparison(results, save_path=timing_path)


if __name__ == "__main__":
    app()
