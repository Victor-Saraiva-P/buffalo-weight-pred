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
        ...

    def feature_selection_dependencies(self) -> dict[str, str]:
        """Report packages; for example, the mapping includes PyTorch."""
        ...

    def repository_commit(self) -> str:
        """Report source identity; for example, this returns the checkout SHA."""
        ...


class SystemFeatureSelectionProvenance:
    """Discover selection identity; for example, production manifests use this adapter."""

    def feature_selection_recipe_hash(self) -> str:
        """Hash selection knowledge; for example, unrelated legacy files are excluded."""
        source_root = Path(__file__).parent
        digest = hashlib.sha256()
        for name in _selection_module_names():
            path = source_root / name
            digest.update(name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def feature_selection_dependencies(self) -> dict[str, str]:
        """Report packages; for example, manifests pin Matplotlib and PyTorch."""
        distributions = ("numpy", "scipy", "scikit-learn", "matplotlib", "torch")
        return {name: importlib.metadata.version(name) for name in distributions}

    def repository_commit(self) -> str:
        """Report source identity; for example, this returns the checkout's full Git SHA."""
        root = Path(__file__).parents[2]
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()


def _selection_module_names() -> tuple[str, ...]:
    return (
        "dense_feature_adapter.py", "feature_baselines.py", "feature_evaluation.py",
        "feature_recommendations.py", "feature_redundancy.py",
        "feature_selection_contract.py", "feature_selection_io.py",
        "feature_selection_manifest.py", "feature_selection_plots.py",
        "feature_selection_provenance.py", "feature_selection_report.py",
        "feature_selection_rules.py", "feature_selection_stage.py",
        "feature_selection_validation.py", "report_cli.py", "snapshot_io.py",
    )
