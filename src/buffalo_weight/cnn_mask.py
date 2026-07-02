from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from buffalo_weight.split import parse_weight


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def find_mask_path(masks_dir: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        path = masks_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(stem)


def load_mask(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    values = np.unique(np.asarray(image))
    invalid_values = [int(value) for value in values if value not in (0, 255)]
    if invalid_values:
        preview = ", ".join(str(value) for value in invalid_values[:10])
        raise ValueError(f"mask must be binary black/white (0/255): {path}; found values: {preview}")
    image = image.resize((image_size, image_size), Image.Resampling.NEAREST)
    return (np.asarray(image, dtype=np.float32) > 0).astype(np.float32)


class CnnMaskRegressor:
    def __init__(self, masks_dir: Path, params: dict[str, Any]) -> None:
        self.masks_dir = masks_dir
        self.image_size = int(params["image_size"])
        self.epochs = int(params["epochs"])
        self.batch_size = int(params["batch_size"])
        self.learning_rate = float(params["learning_rate"])
        self.random_state = int(params["random_state"])
        self.weight_decay = float(params.get("weight_decay", 0.0))
        self.patience = int(params.get("patience", 0))
        self.augment = bool(params.get("augment", False))
        self.model = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, rows: list[dict[str, str]], validation_rows: list[dict[str, str]] | None = None) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as error:
            raise ValueError(
                "cnn_mask requires PyTorch. Install torch in the project environment before using model: cnn_mask"
            ) from error

        torch.manual_seed(self.random_state)
        x = np.stack(
            [load_mask(find_mask_path(self.masks_dir, row["file_name"]), self.image_size) for row in rows]
        )[:, None, :, :]
        y = np.asarray([parse_weight(row["weight"], row["file_name"]) for row in rows], dtype=np.float32)
        self.y_mean = float(y.mean())
        self.y_std = float(y.std() or 1.0)
        y_scaled = (y - self.y_mean) / self.y_std

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y_scaled[:, None])
        validation_x_tensor = validation_y_tensor = None
        if validation_rows:
            validation_x = np.stack(
                [
                    load_mask(find_mask_path(self.masks_dir, row["file_name"]), self.image_size)
                    for row in validation_rows
                ]
            )[:, None, :, :]
            validation_y = np.asarray(
                [parse_weight(row["weight"], row["file_name"]) for row in validation_rows], dtype=np.float32
            )
            validation_y_scaled = (validation_y - self.y_mean) / self.y_std
            validation_x_tensor = torch.from_numpy(validation_x)
            validation_y_tensor = torch.from_numpy(validation_y_scaled[:, None])
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.L1Loss()

        best_loss = float("inf")
        stale_epochs = 0
        generator = torch.Generator().manual_seed(self.random_state)
        for _ in range(self.epochs):
            order = torch.randperm(len(rows), generator=generator)
            epoch_loss = 0.0
            self.model.train()
            for start in range(0, len(rows), self.batch_size):
                indexes = order[start : start + self.batch_size]
                batch_x = x_tensor[indexes]
                if self.augment:
                    batch_x = augment_batch(batch_x, generator)
                prediction = self.model(batch_x)
                loss = loss_fn(prediction, y_tensor[indexes])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(indexes)
            epoch_loss /= len(rows)
            if self.patience <= 0:
                continue
            monitored_loss = epoch_loss
            if validation_x_tensor is not None and validation_y_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    monitored_loss = float(loss_fn(self.model(validation_x_tensor), validation_y_tensor).item())
            if monitored_loss < best_loss:
                best_loss = monitored_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

    def predict(self, rows: list[dict[str, str]]) -> np.ndarray:
        if self.model is None:
            raise ValueError("cnn_mask model has not been fitted")
        import torch

        x = np.stack(
            [load_mask(find_mask_path(self.masks_dir, row["file_name"]), self.image_size) for row in rows]
        )[:, None, :, :]
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(torch.from_numpy(x)).numpy().reshape(-1)
        return prediction * self.y_std + self.y_mean


def augment_batch(batch, generator):
    import torch

    augmented = batch.clone()
    if torch.rand((), generator=generator) < 0.5:
        augmented = torch.flip(augmented, dims=[3])
    shift_y = int(torch.randint(-4, 5, (), generator=generator).item())
    shift_x = int(torch.randint(-4, 5, (), generator=generator).item())
    if shift_y or shift_x:
        augmented = torch.roll(augmented, shifts=(shift_y, shift_x), dims=(2, 3))
        if shift_y > 0:
            augmented[:, :, :shift_y, :] = 0
        elif shift_y < 0:
            augmented[:, :, shift_y:, :] = 0
        if shift_x > 0:
            augmented[:, :, :, :shift_x] = 0
        elif shift_x < 0:
            augmented[:, :, :, shift_x:] = 0
    return augmented
