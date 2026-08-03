"""Selective provenance for comparative feature evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path
from typing import Protocol


class FeatureSelectionProvenance(Protocol):
    """Selection provenance seam; for example, tests inject fixed hashes."""

    def feature_selection_recipe_hash(self) -> str:
        """Hash stage knowledge; for example, a model recipe edit invalidates reuse."""
        # Implementations own hashing so callers cannot substitute a stale manual version.
        ...

    def feature_selection_dependencies(self) -> dict[str, str]:
        """Report packages; for example, the mapping includes PyTorch."""
        # Implementations resolve exact versions at execution time for auditability.
        ...

    def repository_commit(self) -> str:
        """Report source identity; for example, this returns the checkout SHA."""
        # Implementations resolve source identity only when a manifest is constructed.
        ...


class FeatureSelectionEnvironment(Protocol):
    """External provenance boundary; for example, tests supply a named fake."""

    def read_source(self, path: Path) -> bytes:
        """Read recipe knowledge; for example, one source module returns bytes."""
        # The boundary keeps filesystem access replaceable in repeatable tests.
        ...

    def distribution_version(self, name: str) -> str:
        """Read one package version; for example, `torch` resolves to its installed version."""
        # The boundary avoids importing package metadata inside scientific logic.
        ...

    def repository_commit(self, root: Path) -> str:
        """Read Git identity; for example, a clean checkout returns its HEAD SHA."""
        # The boundary makes the Git subprocess replaceable by a named fake.
        ...


class LocalFeatureSelectionEnvironment:
    """Use local files, packages and Git; for example, production provenance owns this I/O."""

    def read_source(self, path: Path) -> bytes:
        """Read source bytes; for example, recipe hashing consumes one module."""
        source_bytes = path.read_bytes()
        return source_bytes

    def distribution_version(self, name: str) -> str:
        """Read an installed version; for example, `numpy` returns its pinned version."""
        version = importlib.metadata.version(name)
        return version

    def repository_commit(self, root: Path) -> str:
        """Read HEAD; for example, manifests record a full source commit."""
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()


class SystemFeatureSelectionProvenance:
    """Discover selection identity; for example, production manifests use this adapter."""

    def __init__(self, environment: FeatureSelectionEnvironment | None = None) -> None:
        """Inject external I/O; for example, tests provide fixed files, versions and Git."""
        resolved_environment = environment or LocalFeatureSelectionEnvironment()
        self._environment = resolved_environment

    def feature_selection_recipe_hash(self) -> str:
        """Hash selection knowledge; for example, unrelated legacy files are excluded."""
        source_root = Path(__file__).parent
        digest = hashlib.sha256()
        for name in _selection_module_names():
            digest.update(name.encode())
            digest.update(self._environment.read_source(source_root / name))
        return digest.hexdigest()

    def feature_selection_dependencies(self) -> dict[str, str]:
        """Report packages; for example, manifests pin Matplotlib and PyTorch."""
        distributions = ("numpy", "scipy", "scikit-learn", "matplotlib", "Pillow", "torch")
        versions = {name: self._environment.distribution_version(name) for name in distributions}
        return versions

    def repository_commit(self) -> str:
        """Report source identity; for example, this returns the checkout's full Git SHA."""
        root = Path(__file__).parents[2]
        commit = self._environment.repository_commit(root)
        return commit


def _selection_module_names() -> tuple[str, ...]:
    names = (
        "dense_feature_adapter.py", "feature_baselines.py", "feature_evaluation.py",
        "feature_recommendations.py", "feature_redundancy.py",
        "feature_selection_artifacts.py", "feature_selection_contract.py",
        "feature_selection_io.py", "feature_selection_manifest.py",
        "feature_selection_plots.py", "feature_selection_provenance.py",
        "feature_selection_report.py", "feature_selection_rules.py",
        "feature_selection_stage.py", "feature_selection_types.py",
        "feature_selection_validation.py",
        "png_artifact.py", "report_cli.py", "snapshot_io.py",
    )
    return names
