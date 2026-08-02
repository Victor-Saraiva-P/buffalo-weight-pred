from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from buffalo_weight.cnn_mask import EarlyStopping, augment_batch, load_mask_inputs, resolve_device
from buffalo_weight.cnn_mask_geometry_network import build_mask_geometry_network
from buffalo_weight.models import ModelParam
from buffalo_weight.split import parse_weight


NetworkBuilder = Callable[[int, int, str, bool, str], nn.Module]


class CnnMaskGeometryRegressor:
    def __init__(
        self,
        masks_dir: Path,
        params: dict[str, ModelParam],
        requested_device: str = "auto",
        network_builder: NetworkBuilder = build_mask_geometry_network,
    ) -> None:
        self.masks_dir = masks_dir
        self.params = params
        self.requested_device = requested_device
        self.network_builder = network_builder
        self.device = "cpu"
        self.model: nn.Module | None = None
        self.feature_mean = np.asarray([], dtype=np.float32)
        self.feature_std = np.asarray([], dtype=np.float32)
        self.target_mean = 0.0
        self.target_std = 1.0

    def fit(
        self,
        rows: list[dict[str, str]],
        feature_columns: list[str],
        validation_rows: list[dict[str, str]] | None = None,
    ) -> None:
        """Fit mask and geometry branches; for example, ``model.fit(rows, features)``."""
        self._seed_training()
        train_tensors = self._training_tensors(rows, feature_columns)
        validation_tensors = self._validation_tensors(validation_rows, feature_columns)
        self.model = self.network_builder(
            train_tensors[0].shape[1],
            len(feature_columns),
            str(self.params.get("architecture", "residual")),
            bool(self.params.get("pretrained", False)),
            str(self.params.get("fine_tune_mode", "head")),
        ).to(self.device)
        self._optimize(train_tensors, validation_tensors)

    def predict(self, rows: list[dict[str, str]], feature_columns: list[str]) -> np.ndarray:
        """Predict one weight per row; for example, ``model.predict(rows, features)``."""
        if self.model is None:
            raise ValueError("cnn_mask_geometry model was not fitted; expected fit before predict")
        masks = torch.from_numpy(self._load_masks(rows)).to(self.device)
        features = torch.from_numpy(self._scaled_features(rows, feature_columns)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            scaled = self.model(masks, features).cpu().numpy().reshape(-1)
        return scaled * self.target_std + self.target_mean

    def _seed_training(self) -> None:
        self.device = resolve_device(self.requested_device, torch.cuda.is_available)
        seed = int(self.params["random_state"])
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)

    def _training_tensors(
        self, rows: list[dict[str, str]], feature_columns: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = _feature_array(rows, feature_columns)
        self.feature_mean = features.mean(axis=0)
        self.feature_std = _safe_standard_deviation(features)
        targets = _target_array(rows)
        self.target_mean = float(targets.mean())
        self.target_std = float(targets.std() or 1.0)
        return self._tensor_triplet(rows, features, targets)

    def _validation_tensors(
        self, rows: list[dict[str, str]] | None, feature_columns: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not rows:
            return None
        return self._tensor_triplet(rows, _feature_array(rows, feature_columns), _target_array(rows))

    def _tensor_triplet(
        self, rows: list[dict[str, str]], features: np.ndarray, targets: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = torch.from_numpy(self._load_masks(rows))
        scaled_features = torch.from_numpy((features - self.feature_mean) / self.feature_std)
        scaled_targets = torch.from_numpy(((targets - self.target_mean) / self.target_std)[:, None])
        return masks, scaled_features, scaled_targets

    def _load_masks(self, rows: list[dict[str, str]]) -> np.ndarray:
        return load_mask_inputs(
            self.masks_dir,
            rows,
            int(self.params["image_size"]),
            str(self.params.get("resize_mode", "letterbox")),
            str(self.params.get("input_representation", "binary")),
        )

    def _scaled_features(
        self, rows: list[dict[str, str]], feature_columns: list[str]
    ) -> np.ndarray:
        return (_feature_array(rows, feature_columns) - self.feature_mean) / self.feature_std

    def _optimize(
        self,
        train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        validation_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> None:
        if self.model is None:
            raise ValueError("cnn_mask_geometry network was missing; expected an initialized network")
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.params["learning_rate"]),
            weight_decay=float(self.params.get("weight_decay", 0.0)),
        )
        stopper = EarlyStopping(int(self.params.get("patience", 0)))
        generator = torch.Generator().manual_seed(int(self.params["random_state"]))
        self._training_epochs(train_tensors, validation_tensors, optimizer, stopper, generator)
        if int(self.params.get("patience", 0)) > 0:
            stopper.restore(self.model)

    def _training_epochs(
        self,
        train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        validation_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        optimizer: torch.optim.Optimizer,
        stopper: EarlyStopping,
        generator: torch.Generator,
    ) -> None:
        for _ in range(int(self.params["epochs"])):
            training_loss = self._training_epoch(train_tensors, optimizer, generator)
            monitored_loss = self._monitored_loss(training_loss, validation_tensors)
            if int(self.params.get("patience", 0)) > 0 and stopper.observe(self.model, monitored_loss):
                break

    def _training_epoch(
        self,
        tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        generator: torch.Generator,
    ) -> float:
        masks, features, targets = tensors
        order = torch.randperm(len(targets), generator=generator)
        total_loss = 0.0
        self.model.train()
        for start in range(0, len(targets), int(self.params["batch_size"])):
            indexes = order[start : start + int(self.params["batch_size"])]
            total_loss += self._training_batch(masks, features, targets, indexes, optimizer, generator)
        return total_loss / len(targets)

    def _training_batch(
        self,
        masks: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor,
        indexes: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        generator: torch.Generator,
    ) -> float:
        batch_masks = masks[indexes]
        if bool(self.params.get("augment", False)):
            batch_masks = augment_batch(batch_masks, generator)
        prediction = self.model(batch_masks.to(self.device), features[indexes].to(self.device))
        loss = nn.functional.l1_loss(prediction, targets[indexes].to(self.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item()) * len(indexes)

    def _monitored_loss(
        self,
        training_loss: float,
        validation_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> float:
        if validation_tensors is None:
            return training_loss
        masks, features, targets = (tensor.to(self.device) for tensor in validation_tensors)
        self.model.eval()
        with torch.no_grad():
            return float(nn.functional.l1_loss(self.model(masks, features), targets).item())


def _feature_array(rows: list[dict[str, str]], feature_columns: list[str]) -> np.ndarray:
    return np.asarray(
        [[_feature_value(row, column) for column in feature_columns] for row in rows], dtype=np.float32
    )


def _feature_value(row: dict[str, str], column: str) -> float:
    raw_value = row.get(column, "")
    try:
        value = float(raw_value.replace(",", "."))
    except ValueError as error:
        raise ValueError(
            f"feature {column} for {row.get('file_name', '')} was {raw_value!r}; expected a finite number"
        ) from error
    if np.isfinite(value):
        return value
    raise ValueError(
        f"feature {column} for {row.get('file_name', '')} was {raw_value!r}; expected a finite number"
    )


def _target_array(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [parse_weight(row["weight"], row.get("file_name", "")) for row in rows], dtype=np.float32
    )


def _safe_standard_deviation(features: np.ndarray) -> np.ndarray:
    standard_deviation = features.std(axis=0)
    standard_deviation[standard_deviation == 0.0] = 1.0
    return standard_deviation
