from typer import Typer
from pathlib import Path

APP = Typer()

@APP.command()
def kaggle(test_data: Path = Path("data/kaggle/test.t"), train_data: Path = Path("data/kaggle")):
    """Test the QCI connection."""
    