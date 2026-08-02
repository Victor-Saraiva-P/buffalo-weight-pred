from __future__ import annotations

import torch
from torch import nn


MASK_GEOMETRY_ARCHITECTURES = frozenset({"residual", "resnet18"})


class MaskGeometryNetwork(nn.Module):
    def __init__(self, input_channels: int, feature_count: int) -> None:
        super().__init__()
        self.mask_encoder = _mask_encoder(input_channels)
        self.geometry_encoder = _geometry_encoder(feature_count)
        self.regression_head = nn.Sequential(
            nn.Linear(160, 64), nn.ReLU(), nn.Dropout(0.25), nn.Linear(64, 1)
        )

    def forward(self, masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        mask_embedding = self.mask_encoder(masks).flatten(1)
        geometry_embedding = self.geometry_encoder(features)
        return self.regression_head(torch.cat((mask_embedding, geometry_embedding), dim=1))


class ResNetMaskGeometryNetwork(nn.Module):
    def __init__(self, feature_count: int, pretrained: bool, fine_tune_mode: str) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.mask_encoder = resnet18(weights=weights)
        self.mask_encoder.fc = nn.Identity()
        self.geometry_encoder = _geometry_encoder(feature_count)
        self.regression_head = nn.Sequential(
            nn.Linear(544, 64), nn.ReLU(), nn.Dropout(0.25), nn.Linear(64, 1)
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        _configure_resnet_training(self.mask_encoder, self.regression_head, fine_tune_mode)

    def train(self, mode: bool = True) -> ResNetMaskGeometryNetwork:
        super().train(mode)
        if mode:
            for module in self.mask_encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        rgb_masks = masks.repeat(1, 3, 1, 1) if masks.shape[1] == 1 else masks
        mask_embedding = self.mask_encoder((rgb_masks - self.image_mean) / self.image_std)
        geometry_embedding = self.geometry_encoder(features)
        return self.regression_head(torch.cat((mask_embedding, geometry_embedding), dim=1))


def _mask_encoder(input_channels: int) -> nn.Module:
    return nn.Sequential(
        _convolution_block(input_channels, 16, 2),
        _convolution_block(16, 32, 2),
        _convolution_block(32, 64, 2),
        nn.AdaptiveAvgPool2d((2, 1)),
    )


def _convolution_block(input_channels: int, output_channels: int, stride: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, output_channels),
        nn.ReLU(),
    )


def _geometry_encoder(feature_count: int) -> nn.Module:
    return nn.Sequential(nn.Linear(feature_count, 32), nn.ReLU(), nn.Dropout(0.15))


def _configure_resnet_training(backbone: nn.Module, head: nn.Module, fine_tune_mode: str) -> None:
    if fine_tune_mode not in {"head", "last_block"}:
        raise ValueError(
            f"ResNet fine tune mode was {fine_tune_mode!r}; expected one of ['head', 'last_block']"
        )
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
        parameter.requires_grad = True
    if fine_tune_mode == "last_block":
        for parameter in backbone.layer4.parameters():
            parameter.requires_grad = True


def build_mask_geometry_network(
    input_channels: int,
    feature_count: int,
    architecture: str = "residual",
    pretrained: bool = False,
    fine_tune_mode: str = "head",
) -> nn.Module:
    """Build a late-fusion network; for example, ``build_mask_geometry_network(3, 10)``."""
    if architecture == "residual":
        return MaskGeometryNetwork(input_channels, feature_count)
    if architecture == "resnet18":
        return ResNetMaskGeometryNetwork(feature_count, pretrained, fine_tune_mode)
    raise ValueError(
        f"mask geometry architecture was {architecture!r}; expected one of {sorted(MASK_GEOMETRY_ARCHITECTURES)}"
    )
