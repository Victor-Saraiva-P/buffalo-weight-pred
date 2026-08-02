from __future__ import annotations

import unittest
from unittest.mock import patch

from buffalo_weight.cnn_architectures import build_mask_network
from buffalo_weight.cnn_mask import resolve_device
from buffalo_weight.pretrained_mask_embedding import build_embedding_network


class FailingTorchvisionNetworkBuilder:
    def __call__(self, **kwargs: object) -> object:
        raise AssertionError(f"torchvision builder received {kwargs!r}; expected no download path")


class NeuralEnvironmentTest(unittest.TestCase):
    def test_explicit_cpu_is_rejected_for_neural_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "cpu.*expected.*cuda"):
            resolve_device("cpu", lambda: True)

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
