"""Selective source and dependency identity for the ResNet-18 baseline."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path
from typing import Protocol


class ResNetBaselineProvenance(Protocol):
    """Provenance seam; for example, tests inject stable versions and source hashes."""

    def recipe_hash(self) -> str:
        """Hash pertinent source; for example, augmentation edits invalidate reuse."""
        ...

    def dependency_versions(self) -> dict[str, str]:
        """Resolve scientific dependencies; for example, include torch and torchvision."""
        ...

    def repository_commit(self) -> str:
        """Return an audit commit; for example, manifests record the producing checkout."""
        ...


class SystemResNetBaselineProvenance:
    """Read local source, packages and Git; for example, production stages use this."""

    def recipe_hash(self) -> str:
        """Hash pertinent source; for example, unrelated baseline files are excluded."""
        digest = hashlib.sha256()
        source_root = Path(__file__).parent
        for name in _recipe_module_names():
            digest.update(name.encode())
            digest.update((source_root / name).read_bytes())
        return digest.hexdigest()

    def dependency_versions(self) -> dict[str, str]:
        """Resolve exact versions; for example, all mask-training dependencies are listed."""
        names = ("numpy", "Pillow", "scikit-learn", "torch", "torchvision")
        return {name: importlib.metadata.version(name) for name in names}

    def repository_commit(self) -> str:
        """Read HEAD; for example, provenance records the full Git SHA."""
        repository_root = Path(__file__).parents[2]
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()


def _recipe_module_names() -> tuple[str, ...]:
    return (
        "resnet18_weights.py", "resnet_baseline_adapter.py",
        "resnet_baseline_artifacts.py", "resnet_baseline_evaluation.py",
        "resnet_baseline_stage.py",
    )
