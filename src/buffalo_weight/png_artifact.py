"""Shared PNG inspection for artifact manifests and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class PngArtifactSpec:
    width_px: int
    height_px: int
    dpi: int


def read_png_artifact_spec(path: Path) -> PngArtifactSpec:
    """Inspect a PNG; for example, canonical figures report dimensions and rounded DPI."""
    with Image.open(path) as figure:
        raw_dpi = figure.info.get("dpi", (0.0, 0.0))
        width, height = figure.size
    specification = PngArtifactSpec(width, height, round(float(raw_dpi[0])))
    return specification
