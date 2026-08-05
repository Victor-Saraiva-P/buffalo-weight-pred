"""Provenance boundary for baseline configurations."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path
from typing import Protocol


class BaselineProvenance(Protocol):
    """Discover configuration identity; for example, tests provide fixed hashes."""

    def baseline_recipe_hash(self, configuration: str) -> str:
        """Hash recipe knowledge; for example, RF and reference hashes are independent."""
        # Implementations own discovery so callers cannot pass a stale manual version.
        ...

    def baseline_dependencies(self, configuration: str) -> dict[str, str]:
        """Report pertinent packages; for example, RF records scikit-learn."""
        # Resolution at execution time keeps manifests tied to the scientific environment.
        ...

    def repository_commit(self) -> str:
        """Report source identity; for example, a manifest records HEAD."""
        # Git access remains replaceable so tests never depend on the surrounding checkout.
        ...


class SystemBaselineProvenance:
    """Discover local recipe, package and Git identity for baseline manifests."""

    def baseline_recipe_hash(self, configuration: str) -> str:
        """Hash pertinent modules; for example, reference excludes the RF adapter."""
        digest = hashlib.sha256()
        source_root = Path(__file__).parent
        for name in _recipe_module_names(configuration):
            digest.update(name.encode())
            digest.update((source_root / name).read_bytes())
        return digest.hexdigest()

    def baseline_dependencies(self, configuration: str) -> dict[str, str]:
        """Report pertinent packages; for example, RF additionally records scikit-learn."""
        names = ["numpy"]
        if configuration == "random_forest_baseline":
            names.append("scikit-learn")
        return {name: importlib.metadata.version(name) for name in names}

    def repository_commit(self) -> str:
        """Read HEAD; for example, completed manifests preserve their source commit."""
        root = Path(__file__).parents[2]
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()


def _recipe_module_names(configuration: str) -> tuple[str, ...]:
    shared = (
        "baseline_artifacts.py", "baseline_evaluation.py", "baseline_manifest.py",
        "baseline_metrics.py", "baseline_stage.py", "baseline_types.py",
    )
    if configuration == "random_forest_baseline":
        return (*shared, "feature_baselines.py")
    if configuration == "training_mean_reference":
        return shared
    raise ValueError(
        f"baseline configuration was {configuration!r}; expected a frozen configuration name"
    )
