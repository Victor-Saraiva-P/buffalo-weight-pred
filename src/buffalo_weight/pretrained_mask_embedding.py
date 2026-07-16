from __future__ import annotations

from pathlib import Path

import numpy as np

from buffalo_weight.cnn_mask import load_masks, resolve_device
from buffalo_weight.mask_classical import regression_pipeline
from buffalo_weight.models import ModelParam
from buffalo_weight.split import parse_weight


EMBEDDING_ARCHITECTURES = frozenset({"mobilenet_v3_small", "resnet18"})


def build_embedding_network(architecture: str) -> object:
    """Build a frozen ImageNet feature extractor; for example, ``build_embedding_network("resnet18")``."""
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights, mobilenet_v3_small, resnet18

    if architecture == "mobilenet_v3_small":
        network = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        network.classifier = nn.Identity()
        return network
    if architecture == "resnet18":
        network = resnet18(weights=ResNet18_Weights.DEFAULT)
        network.fc = nn.Identity()
        return network
    raise ValueError(
        f"embedding architecture was {architecture!r}; expected one of {sorted(EMBEDDING_ARCHITECTURES)}"
    )


class PretrainedMaskEmbeddingRegressor:
    """Regress weight from frozen ImageNet embeddings of binary masks.

    Example: ``PretrainedMaskEmbeddingRegressor(Path("masks"), params).fit(rows)``.
    """

    def __init__(self, masks_dir: Path, params: dict[str, ModelParam], requested_device: str = "auto") -> None:
        self.masks_dir = masks_dir
        self.params = params
        self.image_size = int(params["image_size"])
        self.resize_mode = str(params.get("resize_mode", "letterbox"))
        self.architecture = str(params["architecture"])
        self.batch_size = int(params.get("batch_size", 16))
        self.requested_device = requested_device
        self.model = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def _embeddings(self, rows: list[dict[str, str]]) -> np.ndarray:
        import torch

        device = resolve_device(self.requested_device, torch.cuda.is_available)
        network = build_embedding_network(self.architecture).to(device).eval()
        masks = load_masks(self.masks_dir, rows, self.image_size, self.resize_mode)[:, None]
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
        batches = []
        with torch.no_grad():
            for start in range(0, len(masks), self.batch_size):
                inputs = torch.from_numpy(masks[start : start + self.batch_size]).to(device).repeat(1, 3, 1, 1)
                batches.append(network((inputs - mean) / std).cpu().numpy())
        return np.concatenate(batches)

    def fit(self, rows: list[dict[str, str]]) -> None:
        embeddings = self._embeddings(rows)
        weights = np.asarray([parse_weight(row["weight"], row["file_name"]) for row in rows])
        self.y_mean = float(weights.mean())
        self.y_std = float(weights.std() or 1.0)
        params = {**self.params, "n_components": min(int(self.params.get("n_components", 0)), len(rows) - 1)}
        self.model = regression_pipeline(params, embeddings.shape[1])
        self.model.fit(embeddings, (weights - self.y_mean) / self.y_std)

    def predict(self, rows: list[dict[str, str]]) -> np.ndarray:
        """Predict one weight per mask; for example, ``model.predict(rows)``."""
        if self.model is None:
            raise ValueError("pretrained_mask_embedding model has not been fitted")
        scaled = self.model.predict(self._embeddings(rows))
        return np.asarray(scaled * self.y_std + self.y_mean, dtype=float)
