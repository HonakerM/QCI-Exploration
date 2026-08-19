"""Print the current QCI account and allocation status."""

import typer
from dotenv import load_dotenv
from common.qci import get_client
from common.logging import get_logger, setup_logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOGGER = get_logger(__name__)


def main():
    """Print the current QCI allocations and access state.

    Returns:
        None: Logs the user-visible status for each allocation.
    """
    load_dotenv()  # pull QCI_TOKEN / QCI_API_URL from .env if present
    setup_logging()

    client = get_client()

    for name, alloc in client.get_allocations()["allocations"].items():
        if not alloc["paid"] and not alloc["metered"]:
            LOGGER.info(f"{name}: No Access")
        if alloc["paid"]:
            if alloc["metered"]:
                LOGGER.info(f"{name}: Metered with {alloc['seconds']} remaining")
            else:
                LOGGER.info(f"{name}: Paid")
        elif alloc["metered"]:
            LOGGER.info(f"{name}: Trail with {alloc['seconds']} remaining")


if __name__ == "__main__":
    typer.run(main)
