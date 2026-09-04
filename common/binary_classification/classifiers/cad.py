"""Adapt the upstream CAD reference implementation to the shared adapter API.

CAD is the semi-supervised anomaly detector from "Semi-Supervised Anomaly
Detection through Denoising-Aware Contrastive Distance Learning". Everything
that constitutes the method — the encoder and bilinear tensor distance layer
(`Model`/`distance`), the contrastive objective (`NCELoss2`) and stage one of
the denoising strategy (`normal_sample_denoising`) — is imported unmodified
from the vendored upstream source in `common/binary_classification/vendor/
cad_upstream/CAD_main.py`, not reimplemented here.

What this module adds is only the glue the upstream script does not provide.
Upstream's `train()` cannot be reused directly: it takes the test set as an
argument, evaluates ROC/AP inside the epoch loop, and returns those two
numbers rather than a fitted model, so there is nothing to call `predict` on
afterwards. The loop below therefore follows upstream's structure step for
step while keeping the model, with these deliberate differences:

* Inductive, not transductive. Upstream builds `X_inference` from train *and*
  test rows, so the reference embedding and the stage-two expansion candidates
  are drawn from the test set too. Here `fit` only ever sees training rows, and
  the reference embedding learned from them is reused at `predict` time.
* The best-loss checkpoint is actually kept. Upstream prints "best model
  saved..." but only records the metrics of that epoch; this adapter snapshots
  the weights so the returned model is the one that scored best.
* A decision threshold is derived. Upstream reports ranking metrics only and
  never produces hard labels, so `contamination` below defines the cut.

Note that the batch-size defaults upstream ships are tuned for datasets far
smaller than a credit-card fraud fold, and its anomaly batch spans every
labeled anomaly. `min_anomaly`/`max_anomaly` are exposed so the pairwise
distance matrix stays a workable size.
"""

from dataclasses import dataclass
from pathlib import Path
import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common.binary_classification.base import (
    ClassifierAdapter,
    ClassifierConfig,
    register_classifier,
)
from common.binary_classification.vendor.cad_upstream import (
    Model,
    NCELoss2,
    ODDataset,
    device,
    normal_sample_denoising,
    setup_seed,
)
from common.logging import get_logger

LOGGER = get_logger(__name__)

# Upstream label encoding, used by NCELoss2 and the loaders.
_RELIABLE_NORMAL = 0
_LABELED_ANOMALY = 1
_UNLABELED = 2
_PSEUDO_ANOMALY = 3


@dataclass
class CADConfig(ClassifierConfig):
    """Store the configuration for the upstream CAD model.

    Fields that default to None follow the rule upstream derives them with, so
    leaving them unset reproduces the reference implementation's behavior.

    Attributes:
        algorithm_name (str): Registry key used to resolve this adapter.
        batch_size (int): Rows per reliable-normal batch.
        learning_rate (float): Adam step size.
        labeled_ratio (float): Share of the training anomalies that stay
            labeled; the rest are left in the unlabeled pool.
        n_epochs (int | None): Training epochs; upstream uses 60 under 50k rows
            and 30 at or above it.
        early_stop (int | None): Epochs without a better loss before stopping;
            upstream uses 40 and 20 on the same split.
        hidden_layer (int | None): Embedding width; upstream picks the first of
            4, 8, 16, 32 that is at least a quarter of the feature count.
        n_channels (int | None): Slices in the bilinear tensor distance layer;
            upstream uses three times the embedding width.
        min_anomaly (int | None): Fewest labeled anomalies mixed into a batch;
            upstream uses half of all of them.
        max_anomaly (int | None): Most labeled anomalies mixed into a batch;
            upstream uses all of them.
        contamination (float): Share of the reliable-normal score distribution
            treated as anomalous, which sets the hard-label threshold. Not an
            upstream parameter — upstream reports ranking metrics only.
        random_state (int): Seed passed to upstream's `setup_seed`.
    """

    algorithm_name = "cad"

    batch_size: int = 128
    learning_rate: float = 0.001
    labeled_ratio: float = 0.05

    n_epochs: int | None = None
    early_stop: int | None = None
    hidden_layer: int | None = None
    n_channels: int | None = None
    min_anomaly: int | None = None
    max_anomaly: int | None = None

    contamination: float = 0.01
    random_state: int = 42

    def to_classifier_config(self) -> dict:
        """Convert the config into keyword arguments for the CAD runner.

        Returns:
            dict: Keyword arguments for the runner constructor.
        """
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "labeled_ratio": self.labeled_ratio,
            "n_epochs": self.n_epochs,
            "early_stop": self.early_stop,
            "hidden_layer": self.hidden_layer,
            "n_channels": self.n_channels,
            "min_anomaly": self.min_anomaly,
            "max_anomaly": self.max_anomaly,
            "contamination": self.contamination,
            "random_state": self.random_state,
        }

    @property
    def display_name(self) -> str:
        """Return the model label used in reports and plots.

        Returns:
            str: User-facing display name.
        """
        return "CAD (upstream reference implementation)"


class CADRunner:
    """Fit upstream's CAD model and score rows with the fitted result."""

    def __init__(
        self,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        labeled_ratio: float = 0.05,
        n_epochs: int | None = None,
        early_stop: int | None = None,
        hidden_layer: int | None = None,
        n_channels: int | None = None,
        min_anomaly: int | None = None,
        max_anomaly: int | None = None,
        contamination: float = 0.01,
        random_state: int = 42,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.labeled_ratio = labeled_ratio
        self.n_epochs = n_epochs
        self.early_stop = early_stop
        self.hidden_layer = hidden_layer
        self.n_channels = n_channels
        self.min_anomaly = min_anomaly
        self.max_anomaly = max_anomaly
        self.contamination = contamination
        self.random_state = random_state

        self.model: Model | None = None
        self.reference: torch.Tensor | None = None
        self.threshold: float = 0.0

    # ------------------------------------------------------------------
    # Upstream's derived hyperparameters
    # ------------------------------------------------------------------
    def _resolve_hyperparameters(self, n_rows: int, n_features: int) -> None:
        """Fill in every unset field with the rule upstream derives it by.

        Args:
            n_rows (int): Number of training rows.
            n_features (int): Number of feature columns.
        """
        if self.hidden_layer is None:
            self.hidden_layer = next(
                (x for x in [4, 8, 16, 32] if x >= n_features / 4), 32
            )
        if self.n_channels is None:
            self.n_channels = 3 * self.hidden_layer
        if self.n_epochs is None:
            self.n_epochs = 60 if n_rows < 50000 else 30
        if self.early_stop is None:
            self.early_stop = 40 if n_rows < 50000 else 20

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_bounds(n_rows: int, chunk_size: int) -> list[tuple[int, int]]:
        """Split a row count into chunks, never leaving a chunk of one row.

        Upstream's `distance.forward` calls `.squeeze()` on its intermediates,
        which drops the batch axis when a chunk holds a single row and makes
        the final `mean(dim=1)` fail. Sizing the chunks so that never happens
        keeps the vendored layer usable as written.

        Args:
            n_rows (int): Total rows to score.
            chunk_size (int): Preferred rows per chunk.

        Returns:
            list[tuple[int, int]]: Start and stop index for each chunk.
        """
        if n_rows < 2:
            raise ValueError(
                "CAD scores at least two rows at a time; upstream's distance "
                f"layer cannot handle a single row, got {n_rows}."
            )
        bounds = [
            (start, min(start + chunk_size, n_rows))
            for start in range(0, n_rows, chunk_size)
        ]
        if bounds[-1][1] - bounds[-1][0] == 1 and len(bounds) > 1:
            previous_start, previous_stop = bounds[-2]
            bounds[-2] = (previous_start, previous_stop - 1)
            bounds[-1] = (previous_stop - 1, n_rows)
        return bounds

    def _score_embeddings(
        self, embeddings: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """Score embeddings against a reference vector with upstream's layer.

        Args:
            embeddings (torch.Tensor): Encoded rows.
            reference (torch.Tensor): Reference embedding to measure against.

        Returns:
            torch.Tensor: One score per row; larger means more anomalous.
        """
        assert self.model is not None
        scores = []
        for start, stop in self._chunk_bounds(len(embeddings), 4096):
            chunk = self.model.distance(
                embeddings[start:stop], reference.unsqueeze(0)
            )
            scores.append(chunk.detach().cpu())
        return torch.cat(scores, dim=-1)

    def _score_rows(self, X: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Encode rows and score them against a reference embedding.

        Args:
            X (torch.Tensor): Feature rows.
            reference (torch.Tensor): Reference embedding to measure against.

        Returns:
            torch.Tensor: One score per row.
        """
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            return self._score_embeddings(self.model.fc(X), reference)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _build_loaders(
        self, labels: np.ndarray, normal_idx: np.ndarray, anomaly_idx: np.ndarray
    ) -> tuple[DataLoader, DataLoader]:
        """Build the reliable-normal and anomaly loaders upstream trains from.

        Args:
            labels (np.ndarray): Upstream-encoded labels for the training rows.
            normal_idx (np.ndarray): Indices of the reliable-normal rows.
            anomaly_idx (np.ndarray): Indices of the labeled anomalies.

        Returns:
            tuple[DataLoader, DataLoader]: Normal loader and anomaly loader.
        """
        dataset = ODDataset(labels)
        normal_loader = DataLoader(
            Subset(dataset, normal_idx),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        anomaly_loader = DataLoader(
            Subset(dataset, anomaly_idx),
            batch_size=len(anomaly_idx),
            shuffle=True,
            pin_memory=True,
        )
        return normal_loader, anomaly_loader

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CADRunner":
        """Train CAD on a labeled set reduced to a semi-supervised one.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels in {-1, 1} form.

        Returns:
            CADRunner: The fitted runner.

        Raises:
            ValueError: If the training set holds no anomalies to label.
        """
        setup_seed(self.random_state)
        n_rows, n_features = X.shape
        self._resolve_hyperparameters(n_rows, n_features)

        binary_labels = (np.asarray(y) == 1).astype(int)
        positive_idx = np.where(binary_labels == 1)[0]
        if len(positive_idx) == 0:
            raise ValueError("CAD needs at least one labeled anomaly to train.")

        # Upstream keeps the first `labeled_ratio` share of the training
        # anomalies as the labeled set and leaves the rest unlabeled.
        n_labeled = max(1, int(self.labeled_ratio * len(positive_idx)))
        anomaly_idx = positive_idx[:n_labeled]

        # Stage 1 - normal sample denoising (upstream, unmodified).
        normal_idx, n_pseudos = normal_sample_denoising(X, binary_labels)

        labels = np.full(n_rows, _UNLABELED)
        labels[normal_idx] = _RELIABLE_NORMAL
        labels[anomaly_idx] = _LABELED_ANOMALY
        LOGGER.info(
            "  CAD pools: %s reliable normal | %s labeled anomaly | %s unlabeled",
            int((labels == _RELIABLE_NORMAL).sum()),
            int((labels == _LABELED_ANOMALY).sum()),
            int((labels == _UNLABELED).sum()),
        )

        if self.max_anomaly is None:
            self.max_anomaly = len(anomaly_idx)
        if self.min_anomaly is None:
            self.min_anomaly = int(len(anomaly_idx) * 0.5)
        self.max_anomaly = min(self.max_anomaly, len(anomaly_idx))
        self.min_anomaly = min(self.min_anomaly, self.max_anomaly)

        normal_loader, anomaly_loader = self._build_loaders(
            labels, normal_idx, anomaly_idx
        )

        self.model = Model(
            n_features, self.hidden_layer, n_channels=self.n_channels
        ).to(device)
        X_tensor = torch.FloatTensor(X).to(device)
        criterion = NCELoss2()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.2
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2, T_mult=2, eta_min=0.01 * self.learning_rate
        )

        best_loss = float("inf")
        best_state = None
        early_stop_count = 0

        for epoch in range(self.n_epochs):
            self.model.train()
            loss_record = []
            for x0_idx, y_batch in normal_loader:
                n_anomaly = random.randint(self.min_anomaly, self.max_anomaly)
                sampled_idx, sampled_y = next(iter(anomaly_loader))
                picked = torch.randperm(sampled_idx.shape[0])[:n_anomaly]
                sampled_idx, sampled_y = sampled_idx[picked], sampled_y[picked]

                x0_idx = torch.cat((sampled_idx, x0_idx), dim=0)
                y_batch = torch.cat((sampled_y, y_batch), dim=0)

                optimizer.zero_grad()
                embeddings = self.model.fc(X_tensor[x0_idx])
                distances = self.model.distance(embeddings, embeddings)
                loss = criterion(distances, y_batch)
                loss.backward()
                optimizer.step()
                loss_record.append(loss.detach().item())
            scheduler.step()

            mean_loss = float(np.mean(loss_record))
            LOGGER.info(
                "  CAD epoch %s/%s loss=%.6f", epoch + 1, self.n_epochs, mean_loss
            )

            # Stage 2 - anomalous sample expansion. Upstream ranks train and
            # test rows together; scoring the training rows alone keeps `fit`
            # from seeing the test fold.
            if epoch > int(self.n_epochs / 2):
                self.model.eval()
                with torch.no_grad():
                    embeddings = self.model.fc(X_tensor)
                    scores = self._score_embeddings(
                        embeddings, embeddings.mean(dim=0)
                    )
                _, expanded = torch.topk(scores, min(n_pseudos, len(scores)))
                expanded_labels = labels.copy()
                expanded_labels[expanded.numpy()] = _PSEUDO_ANOMALY
                normal_loader, anomaly_loader = self._build_loaders(
                    expanded_labels, normal_idx, anomaly_idx
                )

                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                if early_stop_count == self.early_stop:
                    LOGGER.info("  CAD early stop at epoch %s", epoch + 1)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Freeze the reference embedding from the training rows so `predict`
        # stays inductive, then set the hard-label cut from the scores of the
        # rows CAD considers reliably normal.
        self.model.eval()
        with torch.no_grad():
            self.reference = self.model.fc(X_tensor).mean(dim=0).detach()
        train_scores = self._score_rows(X_tensor, self.reference).numpy()
        self.threshold = float(
            np.quantile(train_scores[normal_idx], 1.0 - self.contamination)
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return the learned distance to the reference for each row.

        Args:
            X (np.ndarray): Feature rows to score.

        Returns:
            np.ndarray: Anomaly score per row; larger means more anomalous.
        """
        if self.model is None or self.reference is None:
            raise RuntimeError("CAD must be fitted before scoring.")
        X_tensor = torch.FloatTensor(np.asarray(X)).to(device)
        return self._score_rows(X_tensor, self.reference).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Map the tanh-bounded scores onto [0, 1].

        Args:
            X (np.ndarray): Feature rows to score.

        Returns:
            np.ndarray: Positive-class scores.
        """
        # Upstream's distance layer returns the mean of tanh activations, so
        # scores already live in (-1, 1); this shift is monotone and leaves
        # every ranking metric unchanged.
        return np.clip((self.score_samples(X) + 1.0) / 2.0, 0.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard labels using the calibrated threshold.

        Args:
            X (np.ndarray): Feature rows to score.

        Returns:
            np.ndarray: Predicted labels in {-1, 1} form.
        """
        return np.where(self.score_samples(X) > self.threshold, 1, -1)


@register_classifier
class CADAdapter(ClassifierAdapter[CADConfig]):
    """Wrap the upstream CAD model in the shared adapter API."""

    def __init__(self, config: CADConfig):
        """Create the CAD adapter.

        Args:
            config (CADConfig): CAD configuration.
        """
        super().__init__(config)
        self.model = CADRunner(**config.to_classifier_config())

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train CAD on the provided training rows.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels in {-1, 1} format.
        """
        # No scaler here on purpose: the data config already scales features,
        # and upstream scales in its own preprocessing step rather than in the
        # model, so adding one would scale twice.
        self.model.fit(np.asarray(X_train), np.asarray(y_train))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions in the shared {-1, 1} format.

        Args:
            X (np.ndarray): Feature rows to score.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the positive-class score for each row.

        Args:
            X (np.ndarray): Feature rows to score.

        Returns:
            np.ndarray: Positive-class probabilities.
        """
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        """Persist the fitted weights and calibration to disk.

        Args:
            path (Path): Output file for the serialized model.
        """
        if self.model.model is None:
            raise RuntimeError("CAD must be fitted before saving.")
        torch.save(
            {
                "state_dict": self.model.model.state_dict(),
                "reference": self.model.reference,
                "threshold": self.model.threshold,
                "config": self.config.to_classifier_config(),
                "hidden_layer": self.model.hidden_layer,
                "n_channels": self.model.n_channels,
            },
            str(path),
        )


__all__ = ["CADAdapter", "CADConfig", "CADRunner"]
