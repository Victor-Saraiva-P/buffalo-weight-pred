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

    def recipe_hash_at_commit(self, commit: str) -> str | None:
        """Attest source identity; for example, a forged commit returns no recipe hash."""
        ...


class ResNetBaselineEnvironment(Protocol):
    """External provenance I/O; for example, tests supply in-memory source files."""

    def read_source(self, path: Path) -> bytes:
        """Read working source; for example, recipe identity includes the adapter."""
        ...

    def distribution_version(self, name: str) -> str:
        """Resolve one package version; for example, record the installed torch build."""
        ...

    def repository_commit(self, root: Path) -> str:
        """Resolve HEAD; for example, manifests bind results to a full Git SHA."""
        ...

    def committed_source(self, root: Path, commit: str, relative: str) -> bytes | None:
        """Read committed source; for example, an unknown SHA returns ``None``."""
        ...


class LocalResNetBaselineEnvironment:
    """Perform local provenance I/O; for example, production reads packages and Git."""

    def read_source(self, path: Path) -> bytes:
        """Read one source path; for example, bytes preserve exact recipe identity."""
        return path.read_bytes()

    def distribution_version(self, name: str) -> str:
        """Read installed metadata; for example, torchvision changes invalidate reuse."""
        return importlib.metadata.version(name)

    def repository_commit(self, root: Path) -> str:
        """Read HEAD; for example, audit metadata stores the producing checkout."""
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()

    def committed_source(self, root: Path, commit: str, relative: str) -> bytes | None:
        """Read a Git blob; for example, missing objects cannot attest a manifest."""
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative}"],
            capture_output=True,
        )
        return result.stdout if result.returncode == 0 else None


class SystemResNetBaselineProvenance:
    """Read local source, packages and Git; for example, production stages use this."""

    def __init__(self, environment: ResNetBaselineEnvironment | None = None) -> None:
        self._environment = environment or LocalResNetBaselineEnvironment()

    def recipe_hash(self) -> str:
        """Hash pertinent source; for example, unrelated baseline files are excluded."""
        source_root = Path(__file__).parent
        sources = [
            (name, self._environment.read_source(source_root / name))
            for name in _recipe_module_names()
        ]
        return _source_digest(sources)

    def dependency_versions(self) -> dict[str, str]:
        """Resolve exact versions; for example, all mask-training dependencies are listed."""
        names = ("numpy", "Pillow", "scikit-learn", "torch", "torchvision")
        return {name: self._environment.distribution_version(name) for name in names}

    def repository_commit(self) -> str:
        """Read HEAD; for example, provenance records the full Git SHA."""
        return self._environment.repository_commit(Path(__file__).parents[2])

    def recipe_hash_at_commit(self, commit: str) -> str | None:
        """Hash recipe files from Git; for example, audit commits bind to their source."""
        repository_root = Path(__file__).parents[2]
        sources = []
        for name in _recipe_module_names():
            relative = f"src/buffalo_weight/{name}"
            content = self._environment.committed_source(
                repository_root, commit, relative
            )
            if content is None:
                return None
            sources.append((name, content))
        return _source_digest(sources)


def _recipe_module_names() -> tuple[str, ...]:
    return (
        "resnet18_weights.py", "resnet_baseline_adapter.py",
        "resnet_baseline_artifacts.py", "resnet_baseline_evaluation.py",
        "resnet_baseline_provenance.py", "resnet_baseline_stage.py", "resnet_mask.py",
    )


def _source_digest(sources: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, content in sources:
        digest.update(name.encode())
        digest.update(content)
    return digest.hexdigest()
