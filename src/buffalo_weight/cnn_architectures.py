from __future__ import annotations

import torch
from torch import nn
from collections.abc import Callable


MASK_NETWORK_ARCHITECTURES = frozenset(
    {"baseline", "efficientnet_b0", "mobilenet_v3_small", "residual", "resnet18"}
)
IMAGENET_FINE_TUNE_MODES = frozenset({"head", "last_block"})


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, output_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(8, output_channels),
            )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.convolutions(inputs) + self.shortcut(inputs))


class ResidualMaskNetwork(nn.Module):
    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 16, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(16, 16),
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(inputs))
        average = torch.mean(features, dim=(2, 3))
        maximum = torch.amax(features, dim=(2, 3))
        return self.head(torch.cat((average, maximum), dim=1))


class ImageNetMaskNetwork(nn.Module):
    def __init__(
        self, backbone: nn.Module, head: nn.Module, last_block: nn.Module, fine_tune_mode: str
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self._configure_trainable_parameters(head, last_block, fine_tune_mode)

    def _configure_trainable_parameters(
        self, head: nn.Module, last_block: nn.Module, fine_tune_mode: str
    ) -> None:
        if fine_tune_mode not in IMAGENET_FINE_TUNE_MODES:
            raise ValueError(
                f"ImageNet fine tune mode was {fine_tune_mode!r}; "
                f"expected one of {sorted(IMAGENET_FINE_TUNE_MODES)}"
            )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in head.parameters():
            parameter.requires_grad = True
        if fine_tune_mode == "last_block":
            for parameter in last_block.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True) -> ImageNetMaskNetwork:
        super().train(mode)
        if mode:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rgb_masks = inputs.repeat(1, 3, 1, 1) if inputs.shape[1] == 1 else inputs
        normalized_masks = (rgb_masks - self.image_mean) / self.image_std
        return self.backbone(normalized_masks)


def _mobilenet_v3_mask_network(pretrained: bool, fine_tune_mode: str) -> nn.Module:
    try:
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError as error:
        raise ValueError("mobilenet_v3_small requires torchvision") from error
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    backbone = mobilenet_v3_small(weights=weights)
    output_layer = backbone.classifier[-1]
    backbone.classifier[-1] = nn.Linear(output_layer.in_features, 1)
    return ImageNetMaskNetwork(backbone, backbone.classifier, backbone.features[-2:], fine_tune_mode)


def _efficientnet_b0_mask_network(pretrained: bool, fine_tune_mode: str) -> nn.Module:
    try:
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    except ImportError as error:
        raise ValueError("efficientnet_b0 requires torchvision") from error
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    backbone = efficientnet_b0(weights=weights)
    output_layer = backbone.classifier[-1]
    backbone.classifier[-1] = nn.Linear(output_layer.in_features, 1)
    return ImageNetMaskNetwork(backbone, backbone.classifier, backbone.features[-2:], fine_tune_mode)


def _resnet18_mask_network(pretrained: bool, fine_tune_mode: str) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as error:
        raise ValueError("resnet18 requires torchvision") from error
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    backbone = resnet18(weights=weights)
    output_layer = backbone.fc
    backbone.fc = nn.Linear(output_layer.in_features, 1)
    return ImageNetMaskNetwork(backbone, backbone.fc, backbone.layer4, fine_tune_mode)


def _baseline_mask_network(input_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(input_channels, 16, kernel_size=5, padding=2),
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


def build_mask_network(
    architecture: str,
    pretrained: bool = False,
    fine_tune_mode: str = "head",
    input_channels: int = 1,
) -> nn.Module:
    try:
        builder = MASK_NETWORK_BUILDERS[architecture]
    except KeyError as error:
        raise ValueError(
            f"mask network architecture was {architecture!r}; expected one of {sorted(MASK_NETWORK_ARCHITECTURES)}"
        ) from error
    return builder(pretrained, fine_tune_mode, input_channels)


def _build_baseline_network(pretrained: bool, fine_tune_mode: str, input_channels: int) -> nn.Module:
    return _baseline_mask_network(input_channels)


def _build_efficientnet_network(pretrained: bool, fine_tune_mode: str, input_channels: int) -> nn.Module:
    return _efficientnet_b0_mask_network(pretrained, fine_tune_mode)


def _build_mobilenet_network(pretrained: bool, fine_tune_mode: str, input_channels: int) -> nn.Module:
    return _mobilenet_v3_mask_network(pretrained, fine_tune_mode)


def _build_residual_network(pretrained: bool, fine_tune_mode: str, input_channels: int) -> nn.Module:
    return ResidualMaskNetwork(input_channels)


def _build_resnet_network(pretrained: bool, fine_tune_mode: str, input_channels: int) -> nn.Module:
    return _resnet18_mask_network(pretrained, fine_tune_mode)


MASK_NETWORK_BUILDERS: dict[str, Callable[[bool, str, int], nn.Module]] = {
    "baseline": _build_baseline_network,
    "efficientnet_b0": _build_efficientnet_network,
    "mobilenet_v3_small": _build_mobilenet_network,
    "residual": _build_residual_network,
    "resnet18": _build_resnet_network,
}

MASK_NETWORK_RECIPE_SYMBOLS: dict[str, tuple[str, ...]] = {
    "baseline": ("_build_baseline_network", "_baseline_mask_network"),
    "efficientnet_b0": ("_build_efficientnet_network", "_efficientnet_b0_mask_network"),
    "mobilenet_v3_small": ("_build_mobilenet_network", "_mobilenet_v3_mask_network"),
    "residual": ("_build_residual_network", "ResidualMaskNetwork", "ResidualBlock"),
    "resnet18": ("_build_resnet_network", "_resnet18_mask_network"),
}
