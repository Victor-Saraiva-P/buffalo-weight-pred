from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from buffalo_weight.cnn_architectures import build_mask_network
from buffalo_weight.models import ModelParam
from buffalo_weight.split import parse_weight

if TYPE_CHECKING:
    import torch
    from torch import nn


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
RESIZE_MODES = frozenset({"letterbox", "stretch"})
DEVICES = frozenset({"auto", "cpu", "cuda"})
INPUT_REPRESENTATIONS = frozenset({"binary", "geometry_channels"})


def resolve_device(requested: str, cuda_available: Callable[[], bool]) -> str:
    """Resolve a requested compute device, using CUDA for ``auto`` when available.

    Example: ``resolve_device("auto", lambda: False)`` returns ``"cpu"``.
    """
    if requested not in DEVICES:
        raise ValueError(f"device was {requested!r}; expected one of {sorted(DEVICES)}")
    if requested == "auto":
        return "cuda" if cuda_available() else "cpu"
    if requested == "cuda" and not cuda_available():
        raise ValueError("device was 'cuda', but CUDA is not available; expected an available CUDA device")
    return requested


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.stale_epochs = 0
        self.best_state: dict[str, torch.Tensor] | None = None

    def observe(self, model: nn.Module, loss: float) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.stale_epochs = 0
            self.best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            return False
        self.stale_epochs += 1
        return self.stale_epochs >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def find_mask_path(masks_dir: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        path = masks_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(stem)


def resize_mask(image: Image.Image, image_size: int, resize_mode: str) -> Image.Image:
    if resize_mode == "stretch":
        return image.resize((image_size, image_size), Image.Resampling.NEAREST)
    if resize_mode not in RESIZE_MODES:
        raise ValueError(f"resize mode was {resize_mode!r}; expected one of {sorted(RESIZE_MODES)}")
    scale = min(image_size / image.width, image_size / image.height)
    resized_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(resized_size, Image.Resampling.NEAREST)
    letterboxed = Image.new("L", (image_size, image_size), 0)
    offset = ((image_size - resized.width) // 2, (image_size - resized.height) // 2)
    letterboxed.paste(resized, offset)
    return letterboxed


def load_mask(path: Path, image_size: int, resize_mode: str = "letterbox") -> np.ndarray:
    image = Image.open(path).convert("L")
    values = np.unique(np.asarray(image))
    invalid_values = [int(value) for value in values if value not in (0, 255)]
    if invalid_values:
        preview = ", ".join(str(value) for value in invalid_values[:10])
        raise ValueError(f"mask must be binary black/white (0/255): {path}; found values: {preview}")
    resized = resize_mask(image, image_size, resize_mode)
    return (np.asarray(resized, dtype=np.float32) > 0).astype(np.float32)


def load_masks(
    masks_dir: Path, rows: list[dict[str, str]], image_size: int, resize_mode: str = "letterbox"
) -> np.ndarray:
    return np.stack(
        [load_mask(find_mask_path(masks_dir, row["file_name"]), image_size, resize_mode) for row in rows]
    )


def geometry_channels(masks: np.ndarray) -> np.ndarray:
    """Expand binary masks into mask, edge and distance channels; for example, ``geometry_channels(masks)``."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    channels = []
    for mask in masks:
        edge = mask - binary_erosion(mask).astype(np.float32)
        distance = distance_transform_edt(mask).astype(np.float32)
        distance /= float(distance.max() or 1.0)
        channels.append(np.stack((mask, edge, distance)))
    return np.stack(channels).astype(np.float32)


def load_mask_inputs(
    masks_dir: Path,
    rows: list[dict[str, str]],
    image_size: int,
    resize_mode: str,
    representation: str,
) -> np.ndarray:
    """Load CNN channels from binary masks; for example, ``load_mask_inputs(path, rows, 64, "stretch", "binary")``."""
    masks = load_masks(masks_dir, rows, image_size, resize_mode)
    if representation == "binary":
        return masks[:, None]
    if representation == "geometry_channels":
        return geometry_channels(masks)
    raise ValueError(
        f"input representation was {representation!r}; expected one of {sorted(INPUT_REPRESENTATIONS)}"
    )


class CnnMaskRegressor:
    def __init__(self, masks_dir: Path, params: dict[str, ModelParam], requested_device: str = "auto") -> None:
        self.masks_dir = masks_dir
        self.image_size = int(params["image_size"])
        self.resize_mode = str(params.get("resize_mode", "letterbox"))
        self.architecture = str(params.get("architecture", "baseline"))
        self.pretrained = bool(params.get("pretrained", False))
        self.fine_tune_mode = str(params.get("fine_tune_mode", "head"))
        self.input_representation = str(params.get("input_representation", "binary"))
        self.epochs = int(params["epochs"])
        self.batch_size = int(params["batch_size"])
        self.learning_rate = float(params["learning_rate"])
        self.random_state = int(params["random_state"])
        self.weight_decay = float(params.get("weight_decay", 0.0))
        self.patience = int(params.get("patience", 0))
        self.augment = bool(params.get("augment", False))
        self.requested_device = requested_device
        self.device = "cpu"
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

        self.device = resolve_device(self.requested_device, torch.cuda.is_available)
        torch.manual_seed(self.random_state)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.random_state)
        x = load_mask_inputs(
            self.masks_dir, rows, self.image_size, self.resize_mode, self.input_representation
        )
        y = np.asarray([parse_weight(row["weight"], row["file_name"]) for row in rows], dtype=np.float32)
        self.y_mean = float(y.mean())
        self.y_std = float(y.std() or 1.0)
        y_scaled = (y - self.y_mean) / self.y_std

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y_scaled[:, None])
        validation_x_tensor = validation_y_tensor = None
        if validation_rows:
            validation_x = load_mask_inputs(
                self.masks_dir,
                validation_rows,
                self.image_size,
                self.resize_mode,
                self.input_representation,
            )
            validation_y = np.asarray(
                [parse_weight(row["weight"], row["file_name"]) for row in validation_rows], dtype=np.float32
            )
            validation_y_scaled = (validation_y - self.y_mean) / self.y_std
            validation_x_tensor = torch.from_numpy(validation_x)
            validation_y_tensor = torch.from_numpy(validation_y_scaled[:, None])
        self.model = build_mask_network(
            self.architecture, self.pretrained, self.fine_tune_mode, x.shape[1]
        ).to(self.device)
        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(
            trainable_parameters, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.L1Loss()

        early_stopping = EarlyStopping(self.patience)
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
                batch_x = batch_x.to(self.device)
                batch_y = y_tensor[indexes].to(self.device)
                prediction = self.model(batch_x)
                loss = loss_fn(prediction, batch_y)
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
                    validation_prediction = self.model(validation_x_tensor.to(self.device))
                    monitored_loss = float(loss_fn(validation_prediction, validation_y_tensor.to(self.device)).item())
            if early_stopping.observe(self.model, monitored_loss):
                break
        if self.patience > 0:
            early_stopping.restore(self.model)

    def predict(self, rows: list[dict[str, str]]) -> np.ndarray:
        if self.model is None:
            raise ValueError("cnn_mask model has not been fitted")
        import torch

        x = load_mask_inputs(
            self.masks_dir, rows, self.image_size, self.resize_mode, self.input_representation
        )
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(torch.from_numpy(x).to(self.device)).cpu().numpy().reshape(-1)
        return prediction * self.y_std + self.y_mean


def _translate_mask(mask: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    import torch

    translated = torch.roll(mask, shifts=(shift_y, shift_x), dims=(1, 2))
    if shift_y > 0:
        translated[:, :shift_y, :] = 0
    elif shift_y < 0:
        translated[:, shift_y:, :] = 0
    if shift_x > 0:
        translated[:, :, :shift_x] = 0
    elif shift_x < 0:
        translated[:, :, shift_x:] = 0
    return translated


def _augment_mask(mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    import torch

    augmented = mask.clone()
    if torch.rand((), generator=generator) < 0.5:
        augmented = torch.flip(augmented, dims=[2])
    shift_y = int(torch.randint(-4, 5, (), generator=generator).item())
    shift_x = int(torch.randint(-4, 5, (), generator=generator).item())
    return _translate_mask(augmented, shift_y, shift_x)


def augment_batch(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    import torch

    return torch.stack([_augment_mask(mask, generator) for mask in batch])
