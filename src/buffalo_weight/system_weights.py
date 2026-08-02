"""HTTP adapter for setup-managed ResNet-18 weights."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, ContextManager

from buffalo_weight.report_environment import WeightSetupStatus
from buffalo_weight.resnet18_weights import validate_resnet18_sha256


RESNET18_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
UrlOpen = Callable[..., ContextManager[BinaryIO]]


@dataclass(frozen=True)
class HttpWeightGateway:
    _url_open: UrlOpen

    def ensure_resnet18_weights(
        self, cache_path: Path, expected_sha256: str
    ) -> WeightSetupStatus:
        """Download once and verify every reuse; for example, valid cache stays offline."""
        if cache_path.exists():
            validate_resnet18_sha256(cache_path, expected_sha256)
            return WeightSetupStatus.REUSED
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = cache_path.with_suffix(f"{cache_path.suffix}.part")
        self._download(partial_path)
        validate_resnet18_sha256(partial_path, expected_sha256)
        partial_path.replace(cache_path)
        return WeightSetupStatus.DOWNLOADED

    def _download(self, destination: Path) -> None:
        try:
            with self._url_open(RESNET18_URL, timeout=60) as response:
                with destination.open("wb") as output:
                    shutil.copyfileobj(response, output)
        except OSError as error:
            destination.unlink(missing_ok=True)
            raise ValueError(
                f"ResNet-18 download failed from {RESNET18_URL!r}; "
                "expected an accessible official URL"
            ) from error
