"""CLI for comparing saved model results with ROC and metric plots."""

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


@app.command()
def main(
    results_files: list[Path] = typer.Argument(
        ...,
        help="Paths to saved ModelResults JSON files or folders containing JSON files",
    ),
    save_file: Path | None = None,
):
    """Loads results JSONs and plots ROC curves, metrics, and timing comparisons.

    Args:
        results_files (list[Path]): Paths to saved ModelResults JSON files or folders
            containing JSON files.
        save_file (Path): If provided, save the comparison results to this file.
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

    roc_path = None
    metric_path = None
    timing_path = None
    if save_file:
        save_file.parent.mkdir(parents=True, exist_ok=True)
        roc_path = save_file.parent / (save_file.stem + "_roc.png")
        metric_path = save_file.parent / (save_file.stem + "_metrics.png")
        timing_path = save_file.parent / (save_file.stem + "_timing.png")

    plot_roc_curves(results, save_path=roc_path)
    plot_metric_comparison(results, save_path=metric_path)
    plot_timing_comparison(results, save_path=timing_path)


if __name__ == "__main__":
    app()
