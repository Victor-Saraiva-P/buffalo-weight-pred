from __future__ import annotations

import unittest
from collections.abc import Mapping
from pathlib import Path

import torch

from buffalo_weight.resnet18_weights import load_offline_resnet18


class RecordingResNet18:
    def __init__(self) -> None:
        """Start without a loaded state mapping."""
        super().__init__()
        self.loaded_state: dict[str, torch.Tensor] | None = None

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> object:
        """Record the state mapping passed by the offline loader."""
        self.loaded_state = dict(state_dict)
        return object()


class FakeResNet18Factory:
    def __init__(self) -> None:
        """Expose one recording network instance."""

        self.network = RecordingResNet18()

    def create_without_weights(self) -> RecordingResNet18:
        """Return a network without downloading weights."""

        return self.network


class FakeResNet18StateReader:
    def __init__(self) -> None:
        """Start without a requested path or hash."""
        self.path: Path | None = None
        self.expected_sha256: str | None = None

    def read(self, path: Path, expected_sha256: str) -> dict[str, torch.Tensor]:
        self.path = path
        self.expected_sha256 = expected_sha256
        return {"layer.weight": torch.tensor([1.0])}


class ResNet18WeightsTest(unittest.TestCase):
    def test_offline_loader_builds_without_download_and_loads_validated_cache(self) -> None:
        factory = FakeResNet18Factory()
        reader = FakeResNet18StateReader()
        cache_path = Path("generated/setup/resnet18-IMAGENET1K_V1.pth")
        expected_sha256 = "approved-sha"

        network = load_offline_resnet18(factory, reader, cache_path, expected_sha256)

        self.assertIs(network, factory.network)
        self.assertEqual(reader.path, cache_path)
        self.assertEqual(reader.expected_sha256, expected_sha256)
        self.assertIsNotNone(factory.network.loaded_state)
        loaded_state = factory.network.loaded_state
        assert loaded_state is not None
        self.assertEqual(loaded_state["layer.weight"].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
