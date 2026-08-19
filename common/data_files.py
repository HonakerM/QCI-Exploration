"""Helpers for loading and saving YAML and dataclass-backed config files."""

from pathlib import Path
from typing import Type, TypeVar

from dataclass_wizard import asdict, fromdict
import yaml


def load_yaml(file_path: Path) -> dict:
    """Read a YAML file and return its parsed contents.

    Args:
        file_path (Path): YAML file to read.

    Returns:
        dict: Parsed YAML data.
    """
    with file_path.open("r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data


T = TypeVar("T")


def load_data_dict(obj: dict, cls: Type[T]) -> T:
    """Build a dataclass instance from a dictionary payload.

    Args:
        obj (dict): Dictionary containing the dataclass fields.
        cls (Type[T]): Dataclass type to construct.

    Returns:
        T: Dataclass instance populated from the dictionary.
    """
    return fromdict(cls, obj)


def load_data_file(file_path: Path, cls: Type[T]) -> T:
    """Load a YAML file and convert it into the requested dataclass.

    Args:
        file_path (Path): YAML file to read.
        cls (Type[T]): Dataclass type to create.

    Returns:
        T: Dataclass loaded from the YAML content.
    """
    with file_path.open("r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    return fromdict(cls, yaml_data)


def save_data_file(config_obj: T, file_path: Path):
    """Serialize a dataclass to YAML and write it to disk.

    Args:
        config_obj (T): Dataclass instance to save.
        file_path (Path): Output path for the YAML file.
    """
    data_dict = asdict(config_obj)

    with file_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data_dict, file, default_flow_style=False, sort_keys=True)


def convert_path_to_results(path: Path) -> Path:
    """Map a test-definition path to the results JSON path.

    Args:
        path (Path): Input path from the tests directory.

    Returns:
        Path: Matching results file path.
    """
    return Path(*("results" if p == "tests" else p for p in path.parts)).with_suffix(
        ".json"
    )
