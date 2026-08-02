"""Official execution policy for neural prediction paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


OFFICIAL_NEURAL_DEVICE = "cuda"
OFFICIAL_NEURAL_DEVICE_CHOICES = (OFFICIAL_NEURAL_DEVICE,)
SETUP_MANAGED_PRETRAINED_ARCHITECTURES = frozenset({"resnet18"})


class ComputeAvailability(Protocol):
    gpu_name: str | None
    cuda_capability: str | None


def require_neural_cuda(compute: ComputeAvailability) -> None:
    """Reject a CPU-only neural run; for example, setup may pass its compute record."""
    if compute.gpu_name is not None and compute.cuda_capability is not None:
        return
    raise ValueError(
        f"CUDA environment was {compute!r}; "
        "expected an available CUDA GPU with compute capability"
    )


def resolve_neural_device(requested: str, cuda_available: Callable[[], bool]) -> str:
    """Require CUDA for neural work; for example, an available CUDA request succeeds."""
    if requested != OFFICIAL_NEURAL_DEVICE:
        raise ValueError(
            f"neural device was {requested!r}; expected {OFFICIAL_NEURAL_DEVICE!r}"
        )
    if not cuda_available():
        raise ValueError(
            "neural device was 'cuda', but CUDA is not available; "
            "expected an available CUDA device"
        )
    return OFFICIAL_NEURAL_DEVICE


def require_setup_managed_pretraining(architecture: str, pretrained: bool) -> None:
    """Block unmanaged pretrained weights; for example, random MobileNet remains allowed."""
    if not pretrained or architecture in SETUP_MANAGED_PRETRAINED_ARCHITECTURES:
        return
    raise ValueError(
        f"pretrained architecture was {architecture!r}; expected setup-managed "
        f"pretraining from {sorted(SETUP_MANAGED_PRETRAINED_ARCHITECTURES)!r}"
    )
