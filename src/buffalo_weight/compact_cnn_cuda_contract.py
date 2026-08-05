"""Minimal CUDA/checkpoint contract probe for the compact CNN adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn

from buffalo_weight.compact_cnn_adapter import CompactCnnAdapter, CudaRuntime
from buffalo_weight.compact_cnn_network import CompactCnnNetwork


@dataclass(frozen=True)
class CompactCnnContractProbe:
    device_type: str
    loss: float
    has_gradients: bool
    parameters_updated: bool
    predictions: tuple[float, ...]
    model: CompactCnnNetwork


class CompactCnnContractAdapter(CompactCnnAdapter):
    """Exercise the production adapter; for example, CUDA tests verify one update."""

    def __init__(self, runtime: CudaRuntime | None = None) -> None:
        """Initialize production CUDA behavior; for example, missing CUDA still fails early."""
        super().__init__(runtime)
        return None

    def contract_probe(self, seed: int = 44) -> CompactCnnContractProbe:
        """Exercise CUDA behavior; for example, tests perform one tiny parameter update."""
        model = self.create_model(seed)
        inputs = _compact_cnn_probe_inputs(self.device)
        targets = torch.tensor([-0.5, 0.5], device=self.device)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss = nn.functional.l1_loss(model(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        gradients = all(parameter.grad is not None for parameter in model.parameters())
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        updated = any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
        predictions = _probe_predictions(model, inputs)
        return CompactCnnContractProbe("cuda", float(loss.item()), gradients, updated,
                                       predictions, model)

    def save_model(self, model: CompactCnnNetwork, path: Path) -> None:
        """Save an owned checkpoint; for example, CUDA tests reload the same weights."""
        payload = {"state_dict": model.state_dict()}
        torch.save(payload, path)
        return None

    def load_model(self, path: Path, seed: int = 44) -> CompactCnnNetwork:
        """Load an owned checkpoint; for example, tensors remain on CUDA."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
            raise ValueError(
                f"checkpoint payload was {type(payload).__name__}; expected state_dict"
            )
        model = self.create_model(seed)
        model.load_state_dict(cast(dict[str, torch.Tensor], payload["state_dict"]))
        return model

    def probe_predictions(self, model: CompactCnnNetwork) -> tuple[float, ...]:
        """Predict the fixed CUDA probe; for example, checkpoint tests compare values."""
        inputs = _compact_cnn_probe_inputs(self.device)
        predictions = _probe_predictions(model, inputs)
        return predictions


def _compact_cnn_probe_inputs(device: torch.device) -> torch.Tensor:
    inputs = torch.zeros((2, 1, 32, 32), device=device)
    inputs[:, :, 8:24, 6:26] = 1.0
    return inputs


def _probe_predictions(model: CompactCnnNetwork, inputs: torch.Tensor) -> tuple[float, ...]:
    model.eval()
    with torch.no_grad():
        values = model(inputs).detach().cpu().tolist()
    predictions = tuple(float(value) for value in values)
    return predictions
