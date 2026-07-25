from pathlib import Path
from typing import Type, TypeVar

from dataclass_wizard import asdict, fromdict
import yaml


def load_yaml(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data


T = TypeVar("T")


def load_data_dict(obj: dict, cls: Type[T]) -> T:
    """Converts a dictionary to a dataclass instance of the specified type.

    Args:
        obj (dict): The dictionary to convert.
        cls (Type[T]): The dataclass type to convert to.

    Returns:
        T: An instance of the specified dataclass type.
    """
    return fromdict(cls, obj)


def load_data_file(file_path: Path, cls: Type[T]) -> T:
    with file_path.open("r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    return fromdict(cls, yaml_data)


def save_data_file(config_obj: T, file_path: Path):
    """Converts a dataclass instance to a dict and saves it as a YAML file."""
    # 1. Convert the dataclass to a standard Python dictionary
    # asdict() preserves nested dataclasses and resolves types
    data_dict = asdict(config_obj)

    # 2. Write the dictionary to the file using PyYAML

    with file_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data_dict, file, default_flow_style=False, sort_keys=True)


def convert_path_to_results(path: Path) -> Path:
    return Path(*("results" if p == "tests" else p for p in path.parts)).with_suffix(
        ".json"
    )
