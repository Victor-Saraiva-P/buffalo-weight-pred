"""Offline access to the setup-validated ResNet-18 weights."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from buffalo_weight.environment_contract import RESNET18_CACHE_PATH, RESNET18_SHA256

if TYPE_CHECKING:
    from torch import Tensor, nn


class ResNet18Network(Protocol):
    def load_state_dict(self, state_dict: Mapping[str, Tensor]) -> object:
        """Load validated parameters.

        Example: accept the official state mapping without contacting a URL.
        """
        ...


class ResNet18Factory(Protocol):
    def create_without_weights(self) -> ResNet18Network:
        """Build without network access.

        Example: pass ``weights=None`` to torchvision.
        """
        ...


class ResNet18StateReader(Protocol):
    def read(self, path: Path, expected_sha256: str) -> Mapping[str, Tensor]:
        """Read validated local state.

        Example: reject a cache with the wrong SHA-256.
        """
        ...


class TorchvisionResNet18Factory:
    def create_without_weights(self) -> ResNet18Network:
        """Create an empty ResNet-18; for example, never ask torchvision for a URL weight."""
        from torchvision.models import resnet18

        return cast(ResNet18Network, resnet18(weights=None))


class TorchResNet18StateReader:
    def read(self, path: Path, expected_sha256: str) -> Mapping[str, Tensor]:
        """Load a verified torch state mapping; for example, read the setup cache on CPU."""
        import torch

        require_offline_resnet18_weights(path, expected_sha256)
        loaded_state: object = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(loaded_state, dict):
            raise ValueError(
                f"ResNet-18 state type was {type(loaded_state).__name__!r}; expected a mapping"
            )
        return cast("Mapping[str, Tensor]", loaded_state)


def build_offline_resnet18(
    factory: ResNet18Factory,
    state_reader: ResNet18StateReader,
    cache_path: Path = RESNET18_CACHE_PATH, expected_sha256: str = RESNET18_SHA256
) -> nn.Module:
    """Build with injected adapters; for example, tests can supply an in-memory reader."""
    network = factory.create_without_weights()
    state_dict = state_reader.read(cache_path, expected_sha256)
    network.load_state_dict(state_dict)
    from torch import nn

    return cast("nn.Module", network)


def default_offline_resnet18(
    cache_path: Path = RESNET18_CACHE_PATH, expected_sha256: str = RESNET18_SHA256
) -> nn.Module:
    """Compose the system adapters; for example, training reads the setup cache."""
    return build_offline_resnet18(
        TorchvisionResNet18Factory(), TorchResNet18StateReader(), cache_path, expected_sha256
    )


def validate_resnet18_sha256(path: Path, expected_sha256: str) -> None:
    """Verify cache integrity; for example, reject a changed official weight file."""
    actual_sha256 = _file_sha256(path)
    if actual_sha256 == expected_sha256:
        return
    raise ValueError(
        f"ResNet-18 cache SHA-256 was {actual_sha256!r} for {path}; expected {expected_sha256!r}"
    )


def require_offline_resnet18_weights(path: Path, expected_sha256: str) -> None:
    """Require setup-managed weights; for example, a missing cache explains recovery."""
    if not path.exists():
        raise ValueError(
            f"ResNet-18 cache was missing at {path}; expected verified offline weights. "
            "Run `python main.py setup` before the baseline."
        )
    try:
        validate_resnet18_sha256(path, expected_sha256)
    except ValueError as error:
        raise ValueError(
            f"{error} Run `python main.py setup` to restore verified offline weights."
        ) from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
