"""Shared classifier base and registry for binary classification adapters.

This central module exposes `ClassifierConfig`, `ClassifierAdapter`, and the
adapter registry helpers so both `ensemble_classifiers` and `classifiers`
subpackages can reuse a single implementation and stay compatible.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Generic, TypeVar, get_args, get_origin

import numpy as np


class ClassifierConfig(ABC):
    """Hyperparameters for one pluggable classifier.

    Concrete subclasses are expected to be `@dataclass`-decorated and must set
    an `algorithm_name` ClassVar used as the registry key.
    """

    algorithm_name: ClassVar[str]

    @abstractmethod
    def to_classifier_config(self) -> dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def display_name(self) -> str:
        raise NotImplementedError


TConfig = TypeVar("TConfig", bound=ClassifierConfig)


class ClassifierAdapter(ABC, Generic[TConfig]):
    """Wraps a concrete backend model behind a common fit/predict/save API."""

    def __init__(self, config: TConfig):
        self.config: TConfig = config

    @classmethod
    def config_cls(cls) -> type[TConfig]:
        for base in getattr(cls, "__orig_bases__", ()):  # pragma: no cover - typing helper
            if get_origin(base) is ClassifierAdapter:
                args = get_args(base)
                if args and isinstance(args[0], type):
                    return args[0]
        raise TypeError(
            f"{cls.__name__} must parameterize ClassifierAdapter with its config type."
        )

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    def get_train_timing(self) -> dict[str, float] | None:
        return None

    def submission_warning(self) -> str | None:
        return None


# Registry
_REGISTRY: dict[str, type[ClassifierAdapter]] = {}


def register_classifier(adapter_cls: type[ClassifierAdapter]) -> type[ClassifierAdapter]:
    _REGISTRY[adapter_cls.config_cls().algorithm_name] = adapter_cls
    return adapter_cls


def get_adapter_cls(algorithm_name: str) -> type[ClassifierAdapter]:
    try:
        return _REGISTRY[algorithm_name]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}'. Registered algorithms: {available_algorithms()}"
        ) from None


def available_algorithms() -> list[str]:
    return sorted(_REGISTRY)
