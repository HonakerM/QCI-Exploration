"""Run binary-classification experiments defined by YAML config files."""

from pathlib import Path
from dotenv import load_dotenv
import typer

from common.data_files import convert_path_to_results, convert_path_to_results_dir, load_yaml
from common.logging import get_logger, setup_logging


LOGGER = get_logger(__name__)

APP = typer.Typer()


def _results_target_for_yaml(file_path: Path) -> Path:
    """Return the expected result path or directory for a YAML experiment file."""
    data = load_yaml(file_path)
    repetition = data.get("repetition") or data.get("reptition")
    if repetition is not None:
        return convert_path_to_results_dir(file_path)
    return convert_path_to_results(file_path)


@APP.command("test-folder")
def test_folder(
    test_folder: Path,
    dry_run: bool = False,
    display_plots: bool = False,
    suppress_warnings: bool = False,
    rerun: bool = False,
):
    """Run every YAML experiment file in a folder.

    Args:
        test_folder (Path): Directory containing experiment definition files.
        dry_run (bool): Load and validate the configs without training models.
        display_plots (bool): Display any generated plots after each run.
        suppress_warnings (bool): Suppress warnings during the run.
        rerun (bool): Ignore existing result files and rerun the experiments.
    """
    load_dotenv()  # pull QCI_TOKEN / QCI_API_URL / IBM from .env if present
    setup_logging()
    for file in test_folder.glob("**/*.yaml"):
        LOGGER.info("Running test file: %s", file)
        results = _results_target_for_yaml(file)
        if results.is_file() and results.exists() and not rerun:
            LOGGER.warning(
                "Results path %s already exists. Skipping test file %s.",
                results,
                file,
            )
            continue
        test_file(
            file,
            dry_run=dry_run,
            display_plots=display_plots,
            suppress_warnings=suppress_warnings,
            _from_folder=True,
        )


@APP.command("test-file")
def test_file(
    test_file: Path,
    dry_run: bool = False,
    display_plots: bool = True,
    save_plots: bool = False,
    suppress_warnings: bool = False,
    _from_folder: bool = typer.Option(default=False, expose_value=False, hidden=True),
):
    """Run a single experiment from a YAML file.

    Args:
        test_file (Path): YAML file that defines the data and model configuration.
        dry_run (bool): Validate the config and data pipeline without training.
        display_plots (bool): Display training diagnostics and comparison plots.
        suppress_warnings (bool): Suppress warnings during the run.
        save_plots (bool): Save comparison plots to disk when enabled.
    """
    if not _from_folder:
        load_dotenv()  # pull QCI_TOKEN / QCI_API_URL / IBM from .env if present
        setup_logging()

    from common.binary_classification.impl import test_file as impl_test_file

    impl_test_file(
        test_file,
        dry_run,
        display_plots,
        save_plots,
        suppress_warnings=suppress_warnings,
    )


if __name__ == "__main__":
    APP()
