"""Base classes for pluggable, optimization-backed binary classifiers.

Each supported algorithm (QCi's Dirac-3 QBoost today; others to follow)
implements two small pieces against this module:

  * a `ClassifierConfig` — a dataclass of hyperparameters, plus a
    `to_classifier_config()` method that turns them into kwargs for the
    underlying library, and a `display_name` used in results/plots.
  * a `ClassifierAdapter` — a thin wrapper that maps that library's actual
    fit/predict API onto the common interface below.

`train()` in qciboost_fraud.py is written entirely against `ClassifierAdapter`,
so it never needs to know which optimization backend it's driving. Adding a
new algorithm means writing a new Config/Adapter pair and registering it —
no changes to the training/evaluation pipeline itself.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Generic, TypeVar, get_args, get_origin

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ClassifierConfig(ABC):
    """Hyperparameters for one pluggable classifier.

    Concrete subclasses are expected to be `@dataclass`-decorated.
    """

    #: Registry key used to select this classifier, e.g. "cvqboost". Set by
    #: each concrete subclass.
    algorithm_name: ClassVar[str]

    @abstractmethod
    def to_classifier_config(self) -> dict:
        """Converts this config into kwargs consumable by the underlying model.

        Returns:
            dict: Keyword arguments suitable for constructing the backend's
            classifier object.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name used for model naming and plot legends."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

#: Bound to ClassifierConfig so ClassifierAdapter[SomeConfig] gives
#: `self.config` (and the constructor) the concrete SomeConfig type.
TConfig = TypeVar("TConfig", bound=ClassifierConfig)


class ClassifierAdapter(ABC, Generic[TConfig]):
    """Wraps one concrete backend model behind a common fit/predict/save API.

    Subclasses parameterize the generic to pin down their config type, e.g.:

        class CVQBoostAdapter(ClassifierAdapter[CVQBoostConfig]):
            ...

    This gives `self.config` (and anyone constructing the adapter) the exact
    config type, instead of a class-level `config_cls` variable typed as the
    ClassifierConfig base.

    Attributes:
        config (TConfig): Hyperparameters driving this adapter.
    """

    def __init__(self, config: TConfig):
        self.config: TConfig = config

    @classmethod
    def config_cls(cls) -> type[TConfig]:
        """Returns the concrete ClassifierConfig type this adapter was parameterized with.

        Derived from `ClassifierAdapter[SomeConfig]` in the class definition,
        so there's a single source of truth (the type parameter) instead of
        a separately-maintained class variable.

        Returns:
            type[TConfig]: The concrete config class.

        Raises:
            TypeError: If the subclass didn't parameterize the generic, e.g.
                `class Foo(ClassifierAdapter)` instead of
                `class Foo(ClassifierAdapter[FooConfig])`.
        """
        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is ClassifierAdapter:
                args = get_args(base)
                if args and isinstance(args[0], type):
                    return args[0]
        raise TypeError(
            f"{cls.__name__} must parameterize ClassifierAdapter with its "
            f"config type, e.g. class {cls.__name__}(ClassifierAdapter[SomeConfig])."
        )

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fits the underlying model in place.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels in {-1, +1}.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns hard {-1, +1} predictions for X."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns positive-class probabilities in [0, 1] for X."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persists the fitted model to disk at `path`."""
        raise NotImplementedError

    def submission_warning(self) -> str | None:
        """Returns a confirmation message to show before training starts.

        Backends that incur real-world cost or shared-resource usage (e.g.
        QCi's Dirac-3 hardware) should override this to return a warning
        string; the CLI will require the user to type "start" before
        proceeding. Purely local/classical algorithms can leave this as the
        default `None`, which skips the confirmation prompt entirely.
        """
        return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[ClassifierAdapter]] = {}


def register_classifier(
    adapter_cls: type[ClassifierAdapter],
) -> type[ClassifierAdapter]:
    """Class decorator that registers an adapter under its config's algorithm name.

    Usage:
        @register_classifier
        class CVQBoostAdapter(ClassifierAdapter[CVQBoostConfig]):
            ...

    Args:
        adapter_cls (type[ClassifierAdapter]): The adapter class to register.
            Must parameterize ClassifierAdapter with its config type.

    Returns:
        type[ClassifierAdapter]: The same class, unmodified (for decorator use).
    """
    _REGISTRY[adapter_cls.config_cls().algorithm_name] = adapter_cls
    return adapter_cls


def get_adapter_cls(algorithm_name: str) -> type[ClassifierAdapter]:
    """Looks up a registered adapter class by its algorithm name.

    Args:
        algorithm_name (str): Registry key, e.g. "cvqboost".

    Returns:
        type[ClassifierAdapter]: The matching adapter class.

    Raises:
        ValueError: If no adapter is registered under that name.
    """
    try:
        return _REGISTRY[algorithm_name]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}'. "
            f"Registered algorithms: {available_algorithms()}"
        ) from None


def available_algorithms() -> list[str]:
    """Returns the sorted list of currently registered algorithm names."""
    return sorted(_REGISTRY)
