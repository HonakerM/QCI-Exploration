import logging
from pathlib import Path
from typing import Optional


DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Configure root logging with a standard formatter.

    - `level` sets the root logger level (default: `logging.INFO`).
    - if `log_file` is provided a `FileHandler` is added in addition to stdout.
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
    """Return a logger with the given name (convenience wrapper)."""
    return logging.getLogger(name)
