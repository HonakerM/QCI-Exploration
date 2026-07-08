"""CLI for comparing saved model results with ROC and metric plots."""

from pathlib import Path
import typer
from common.binary_classification.data_types import ModelResults
from common.binary_classification.evaluation import print_results
from common.binary_classification.visualization import (
    plot_metric_comparison,
    plot_roc_curves,
)

app = typer.Typer()


@app.command()
def main(
    results_files: list[Path] = typer.Argument(
        ..., help="Paths to saved ModelResults JSON files"
    ),
    save_plots: bool = typer.Option(False, help="Save PNGs instead of showing"),
):
    """Loads results JSONs and plots ROC curves and a metric comparison.

    Args:
        results_files (list[Path]): Paths to saved ModelResults JSON files.
        save_plots (bool): If True, save the plots as PNGs instead of showing them.
    """
    results = [ModelResults.load(f) for f in results_files]

    for r in results:
        print_results(r)

    roc_path = Path("comparison_roc.png") if save_plots else None
    metric_path = Path("comparison_metrics.png") if save_plots else None

    plot_roc_curves(results, save_path=roc_path)
    plot_metric_comparison(results, save_path=metric_path)


if __name__ == "__main__":
    app()
