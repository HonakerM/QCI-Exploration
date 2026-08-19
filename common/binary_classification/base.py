"""Define the shared classifier interfaces and adapter registry."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Generic, TypeVar, get_args, get_origin

import numpy as np


class ClassifierConfig(ABC):
    """Base configuration object for a pluggable classifier.

    Attributes:
        algorithm_name (ClassVar[str]): Registry key used to resolve the adapter.
    """

    algorithm_name: ClassVar[str]

    @abstractmethod
    def to_classifier_config(self) -> dict:
        """Convert the config into the backend model's keyword arguments.

        Returns:
            dict: Keyword arguments for the underlying classifier constructor.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Return the user-facing model name.

        Returns:
            str: Display label for the classifier.
        """
        raise NotImplementedError


TConfig = TypeVar("TConfig", bound=ClassifierConfig)


class ClassifierAdapter(ABC, Generic[TConfig]):
    """Common adapter interface for training, predicting, and saving a model.

    Attributes:
        config (TConfig): Model configuration associated with this adapter.
    """

    def __init__(self, config: TConfig):
        self.config: TConfig = config

    @classmethod
    def config_cls(cls) -> type[TConfig]:
        """Return the configuration dataclass associated with this adapter.

        Returns:
            type[TConfig]: Concrete config class for the adapter.
        """
        seen: set[type] = set()

        def resolve_config(current: type) -> type | None:
            if current in seen:
                return None
            seen.add(current)

            for base in getattr(current, "__orig_bases__", ()):
                origin = get_origin(base)
                if origin is not None:
                    if origin is ClassifierAdapter:
                        args = get_args(base)
                        if args and isinstance(args[0], type):
                            return args[0]
                    if isinstance(origin, type) and issubclass(
                        origin, ClassifierAdapter
                    ):
                        args = get_args(base)
                        if args and isinstance(args[0], type):
                            return args[0]
                        resolved = resolve_config(origin)
                        if resolved is not None:
                            return resolved
                elif isinstance(base, type) and issubclass(base, ClassifierAdapter):
                    resolved = resolve_config(base)
                    if resolved is not None:
                        return resolved

            return None

        resolved = resolve_config(cls)
        if resolved is not None:
            return resolved

        raise TypeError(
            f"{cls.__name__} must parameterize ClassifierAdapter with its config type."
        )

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the wrapped model on the supplied feature matrix.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict hard labels for the provided feature matrix.

        Args:
            X (np.ndarray): Feature rows to score.

        Returns:
            np.ndarray: Predicted labels in the model's normalized {-1, 1} form.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the provided feature matrix.

        Args:
            X (np.ndarray): Feature rows to score.

        Returns:
            np.ndarray: Probability for the positive class.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize the fitted model to disk.

        Args:
            path (Path): Output location for the saved model.
        """
        raise NotImplementedError

    def get_train_timing(self) -> dict[str, float] | None:
        """Return adapter-specific timing values captured during training.

        Returns:
            dict[str, float] | None: Timing breakdown for the adapter, if available.
        """
        return None

    def submission_warning(self) -> str | None:
        """Return any warning that should be shown before submitting a run.

        Returns:
            str | None: Warning text or None when no warning is required.
        """
        return None


# Registry
_REGISTRY: dict[str, type[ClassifierAdapter]] = {}


def register_classifier(
    adapter_cls: type[ClassifierAdapter],
) -> type[ClassifierAdapter]:
    """Register a classifier adapter under its configured algorithm name.

    Args:
        adapter_cls (type[ClassifierAdapter]): Adapter class to register.

    Returns:
        type[ClassifierAdapter]: The registered adapter class.
    """
    _REGISTRY[adapter_cls.config_cls().algorithm_name] = adapter_cls
    return adapter_cls


def get_adapter_cls(algorithm_name: str) -> type[ClassifierAdapter]:
    """Resolve an algorithm name to the registered adapter class.

    Args:
        algorithm_name (str): Algorithm key to look up.

    Returns:
        type[ClassifierAdapter]: Registered adapter class.
    """
    try:
        return _REGISTRY[algorithm_name]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}'. Registered algorithms: {available_algorithms()}"
        ) from None


def available_algorithms() -> list[str]:
    """Return the sorted list of registered algorithm names.

    Returns:
        list[str]: Known algorithm identifiers.
    """
    return sorted(_REGISTRY)
