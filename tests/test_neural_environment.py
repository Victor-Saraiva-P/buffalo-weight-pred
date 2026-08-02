from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from buffalo_weight.cnn_architectures import build_mask_network
from buffalo_weight.cnn_mask import resolve_device
from buffalo_weight.pretrained_mask_embedding import build_embedding_network
from tests.fake_compute import fake_available_cuda


class FailingTorchvisionNetworkBuilder:
    def __call__(self, **kwargs: object) -> object:
        """Fail if rejected pretraining reaches a torchvision download seam."""

        raise AssertionError(f"torchvision builder received {kwargs!r}; expected no download path")


class FakeResNet18Network(nn.Module):
    def __init__(self) -> None:
        """Provide the ResNet attributes consumed by project-owned wrappers."""
        super().__init__()
        self.fc = nn.Linear(3, 1)
        self.layer4 = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one three-value embedding per fake image."""
        pooled = inputs.mean(dim=(2, 3))
        return self.fc(pooled)


class RecordingOfflineResNet18Builder:
    def __init__(self) -> None:
        """Start without building an offline network."""

        self.calls = 0

    def __call__(self) -> nn.Module:
        """Record one build and return a local fake ResNet-18."""
        self.calls += 1
        return FakeResNet18Network()


class NeuralEnvironmentTest(unittest.TestCase):
    def test_mask_network_receives_injected_offline_resnet_builder(self) -> None:
        builder = RecordingOfflineResNet18Builder()

        network = build_mask_network("resnet18", pretrained=True, offline_resnet18_builder=builder)

        self.assertIsInstance(network, nn.Module)
        self.assertEqual(builder.calls, 1)

    def test_embedding_network_receives_injected_offline_resnet_builder(self) -> None:
        builder = RecordingOfflineResNet18Builder()

        network = build_embedding_network("resnet18", builder)

        self.assertIsInstance(network, nn.Module)
        self.assertEqual(builder.calls, 1)

    def test_explicit_cpu_is_rejected_for_neural_execution(self) -> None:
        """Keep CPU fallback outside the official neural execution contract."""

        with self.assertRaisesRegex(ValueError, "cpu.*expected.*cuda"):
            resolve_device("cpu", fake_available_cuda)

    def test_pretrained_mobilenet_is_rejected_before_torchvision_can_download(self) -> None:
        with patch(
            "torchvision.models.mobilenet_v3_small",
            new=FailingTorchvisionNetworkBuilder(),
        ):
            with self.assertRaisesRegex(
                ValueError, "mobilenet_v3_small.*setup-managed.*resnet18"
            ):
                build_mask_network("mobilenet_v3_small", pretrained=True)

    def test_pretrained_efficientnet_is_rejected_before_torchvision_can_download(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "efficientnet_b0.*setup-managed.*resnet18"
        ):
            build_mask_network("efficientnet_b0", pretrained=True)

    def test_pretrained_embedding_is_rejected_before_torchvision_can_download(self) -> None:
        with patch(
            "torchvision.models.mobilenet_v3_small",
            new=FailingTorchvisionNetworkBuilder(),
        ):
            with self.assertRaisesRegex(
                ValueError, "mobilenet_v3_small.*setup-managed.*resnet18"
            ):
                build_embedding_network("mobilenet_v3_small")


if __name__ == "__main__":
    unittest.main()
