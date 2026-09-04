"""Load the vendored upstream CAD reference implementation.

`CAD_main.py` is an unmodified copy of `CAD-main.py` from
https://github.com/JerryGao818/CAD (commit b609f6d). It is a command-line
script: the tail of the file parses `argv` and immediately starts a ten-seed
training run. Importing it directly would therefore run an experiment.

Rather than patch the upstream file — which would make it stop matching
upstream — this module compiles only the part above the entrypoint, so the
model, the distance layer, the contrastive loss and the denoising routine are
importable while none of the script body runs.
"""

import re
from pathlib import Path
from types import ModuleType

# Upstream's argument parser opens the CLI section of the file: everything
# from its definition onward runs a training job on import, everything above
# it is the method itself. The pattern is anchored to the start of a line so
# it matches the definition rather than a mention of it inside a comment.
_ENTRYPOINT_PATTERN = re.compile(r"^def parse_args\(\):", re.MULTILINE)

_SOURCE_PATH = Path(__file__).with_name("CAD_main.py")


def _load_upstream() -> ModuleType:
    """Compile the vendored upstream file up to its command-line entrypoint.

    Returns:
        ModuleType: Module holding the upstream definitions.

    Raises:
        RuntimeError: If the entrypoint is missing or ambiguous, which means
            the vendored copy has drifted and the cut point is unknown.
    """
    source = _SOURCE_PATH.read_text(encoding="utf-8")
    matches = _ENTRYPOINT_PATTERN.findall(source)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one top-level 'parse_args' definition in "
            f"{_SOURCE_PATH}, found {len(matches)}. The vendored CAD copy no "
            "longer matches the expected upstream layout; re-vendor it and "
            "confirm where the script entrypoint begins."
        )
    match = _ENTRYPOINT_PATTERN.search(source)
    assert match is not None
    cut = match.start()

    module = ModuleType("cad_upstream_main")
    module.__file__ = str(_SOURCE_PATH)
    exec(compile(source[:cut], str(_SOURCE_PATH), "exec"), module.__dict__)
    return module


_upstream = _load_upstream()

# Re-export the upstream names the adapter builds on. These objects are the
# paper's method verbatim: the encoder plus bilinear tensor distance layer
# (Model/distance), the contrastive objective (NCELoss2), and stage one of the
# denoising strategy (normal_sample_denoising).
Model = _upstream.Model
distance = _upstream.distance
NCELoss2 = _upstream.NCELoss2
ODDataset = _upstream.ODDataset
normal_sample_denoising = _upstream.normal_sample_denoising
setup_seed = _upstream.setup_seed
device = _upstream.device

UPSTREAM_REPO = "https://github.com/JerryGao818/CAD"
UPSTREAM_COMMIT = "b609f6d9e3849ce18e81e7d6da44923f98c2b12d"

__all__ = [
    "Model",
    "distance",
    "NCELoss2",
    "ODDataset",
    "normal_sample_denoising",
    "setup_seed",
    "device",
    "UPSTREAM_REPO",
    "UPSTREAM_COMMIT",
]
