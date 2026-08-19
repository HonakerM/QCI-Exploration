"""Configure the shared application logger for the project."""

import logging
from pathlib import Path


DEFAULT_FORMAT = "%(asctime)s %(levelname)-4s - %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure the root logger with a standard console and optional file handler.

    Args:
        level (int): Log level to apply to the root logger.
        log_file (Path | None): Optional file path for writing logs in addition to stdout.
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
    """Get a module logger by name.

    Args:
        name (str): Logger name, usually the caller's module name.

    Returns:
        logging.Logger: Logger instance for the requested name.
    """
    return logging.getLogger(name)
