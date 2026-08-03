"""Execution and recipe provenance for reconstructed report stages."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path
from typing import Protocol


class ReportProvenance(Protocol):
    """Provenance seam; for example, tests can provide fixed versions and a commit."""

    def inputs_recipe_hash(self) -> str:
        """Hash stage knowledge.

        Example: a source edit changes this digest.
        """
        ...

    def dependencies(self) -> dict[str, str]:
        """Report packages.

        Example: the mapping includes ``numpy``.
        """
        ...

    def repository_commit(self) -> str:
        """Report source identity.

        Example: this returns a full Git SHA.
        """
        ...


class SystemReportProvenance:
    """System provenance adapter; for example, the inputs stage uses it outside tests."""

    def inputs_recipe_hash(self) -> str:
        """Hash input-stage knowledge; for example, consumed source edits invalidate reuse."""
        source_root = Path(__file__).parent
        modules = [source_root / name for name in _recipe_module_names()]
        calculators = sorted((source_root / "feature_calculators").glob("*.py"))
        return _combined_source_hash(modules + calculators)

    def dependencies(self) -> dict[str, str]:
        """Report packages; for example, the mapping includes the NumPy version."""
        distributions = ("numpy", "Pillow", "scipy", "scikit-learn", "PyYAML")
        return {name: importlib.metadata.version(name) for name in distributions}

    def repository_commit(self) -> str:
        """Report source identity; for example, this returns the checkout's full Git SHA."""
        repository_root = Path(__file__).parents[2]
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()


def _recipe_module_names() -> tuple[str, ...]:
    return (
        "canonical_split.py", "csv_io.py", "curated_inputs.py", "hashing.py",
        "input_schema.py", "inputs_manifest.py", "report_inputs.py",
        "report_provenance.py", "reproduction_config.py", "snapshot_io.py",
    )


def _combined_source_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    source_root = Path(__file__).parent
    for path in paths:
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
