"""Standard logging setup shared across the fraud training scripts."""

import logging
from pathlib import Path


DEFAULT_FORMAT = "%(asctime)s %(levelname)-4s - %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configures root logging with a standard formatter.

    Args:
        level (int): The root logger level.
        log_file (Path | None): If provided, a FileHandler writing to this path is added
            in addition to the stdout handler.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        handlers.append(fh)

    fmt = logging.Formatter(fmt=DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)
    for h in handlers:
        h.setFormatter(fmt)

    root = logging.getLogger()
    # Remove any existing handlers to avoid duplicate logs on repeated setup
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    root.setLevel(level)
    for h in handlers:
        root.addHandler(h)


def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the given name.

    Args:
        name (str): The logger name, typically __name__ of the calling module.

    Returns:
        logging.Logger: The logging.Logger instance for that name.
    """
    return logging.getLogger(name)
